import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from typing import Dict, List, Literal, Optional, Tuple, Union

import rapidjson
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger as _logger
from pydantic import BaseModel, ValidationError

from vivabench.mapper import DeterminsticMapper, LLMMapper
from vivabench.ontology.schema import (
    ClinicalCase,
    InvestigationResult,
    PhysicalExamination,
    Symptom,
)
from vivabench.parser import DeterminsticParser, LLMParser
from vivabench.prompts.examiner import (
    ASSISTANT_BASE_PROMPT,
    ASSISTANT_FULL_INFO_BASE_PROMPT,
    DDX_CONF,
    DDX_SIMPLE,
    ERROR_RETURN_MSG,
)
from vivabench.utils import (
    prettify,
    remove_json_markdown,
    remove_json_markdown_enhanced,
    smart_capitalize,
)

RETRY_LIMIT = 2


class AgentResponse(BaseModel):

    action: Literal[
        "history",
        "examination",
        "imaging",
        "investigation",
        "diagnosis_provisional",
        "diagnosis_final",
    ]
    query: Union[str, list]
    reasoning: Optional[str] = None

    @property
    def full_trace(self):
        return f"Action: {prettify(self.action)}\nQuery: {self.query}\nReasoning: {self.reasoning}"

    @property
    def action_trace(self):
        _action_query = self.model_dump()
        _action_query["reasoning"] = ""
        return json.dumps(_action_query)


class Examiner:

    def __init__(
        self,
        clincase: ClinicalCase,
        examiner_model: BaseChatModel,
        mapper: Literal["deterministic", "llm"] = "llm",
        parser: Literal["deterministic", "llm"] = "llm",
        hx_limit=10,
        phys_limit=5,
        ix_limit=5,
        img_limit=5,
        action_limit=20,
        snomed_embeddings_path="./medical/snomed_embeddings",
        logger=None,
    ):
        self.clincase = clincase
        self.logger = logger or _logger

        if mapper == "deterministic":
            self.mapper = DeterminsticMapper(clincase, snomed_embeddings_path)
        else:
            self.mapper = LLMMapper(clincase, model=examiner_model)

        if parser == "deterministic":
            self.parser = DeterminsticParser(clincase, logger=logger)
        else:
            self.parser = LLMParser(clincase, model=examiner_model, logger=logger)

        self.hx_limit = hx_limit
        self.phys_limit = phys_limit
        self.ix_limit = ix_limit
        self.img_limit = img_limit
        self.action_limit = action_limit

        self.action_count = 0
        self.hx_count = 0
        self.phys_count = 0
        self.ix_count = 0
        self.img_count = 0

        self.reviewed_patient = False

        self.diagnosis_provisional = None
        self.diagnosis_final = None

        self.request_log = []

    def process_response(
        self, agent_response: AgentResponse
    ) -> Tuple[Optional[AgentResponse], str]:
        """Processes a response from an agent and routes relevant actions. This currently simply parses actions, similar to tool use
        Tool use with agents etc. will be a TODO when the ecosystem supports tool calls better
        response (str): Agent response in AgentResponse format

        Returns: Agent Response, Examiner Response Dict[str, str] - Parsed agent action and examiner response to agent
        """

        agent_action = agent_response.action

        if agent_action == "history":
            if not self.reviewed_patient:
                examiner_response = self.process_history(agent_response)
            else:
                examiner_response = "You can no longer review the patient. Please proceed to order any investigations or imaging to help with diagnosis."

        elif agent_action == "examination":
            if not self.reviewed_patient:
                examiner_response = self.process_physical(agent_response)
            else:
                examiner_response = "You can no longer review the patient. Please proceed to order any investigations or imaging to help with diagnosis."

        elif agent_action == "investigation":
            examiner_response = self.process_investigations(agent_response)
            self.reviewed_patient = True
        elif agent_action == "imaging":
            examiner_response = self.process_imaging(agent_response)
            self.reviewed_patient = True
        elif agent_action == "diagnosis_provisional":
            self.diagnosis_provisional = agent_response.query
            self.reviewed_patient = True
            examiner_response = (
                "Thank you. Please proceed to imaging and lab investigations."
            )
        elif agent_action == "diagnosis_final":
            self.diagnosis_final = agent_response.query
            examiner_response = "Final diagnosis was made."
        else:
            raise ValueError(f"Unknown agent action: {agent_action}")

        self.action_count += 1

        if self.action_count == self.action_limit:
            examiner_response += "\nYou have run out of time. Please give your final diagnosis for this patient."

        return agent_response, examiner_response

    def _log_requests(self, query: AgentResponse, requests):

        self.request_log.append(
            {
                "query": query.query,
                "action": query.action,
                "matched": requests.get("matched", []),
                "unmatched": requests.get("unmatched", []),
            }
        )

        for k in requests.get("matched", []):
            self.logger.debug(
                f"Matched: {k['query']} -> {k['key']} {k.get('addit') if k.get('addit') else ''}"
            )
        for k in requests.get("partial", []):
            self.logger.debug(f"Partial: {k['query']} -> {k['key']}")
        for k in requests.get("unmatched", []):
            self.logger.debug(f"Unmatched: {k['query']} -> {k['key']}")

    def process_history(self, query):
        requests = self.mapper.map_history_requests(query)

        self._log_requests(query, requests)
        _prompt = self.parser.parse_history_requests(query, requests)
        self.hx_count += 1
        if self.hx_count >= self.hx_limit:
            _prompt += "\nLimit on history-taking reached. Please proceed to further working up the patient."
        return _prompt

    def process_physical(self, query):

        requests = self.mapper.map_physical_requests(query)

        requests["matched"]

        self._log_requests(query, requests)

        _prompt = self.parser.parse_physical_requests(query, requests)

        self.phys_count += 1
        if self.phys_count >= self.phys_limit:
            _prompt += "\nLimit on physical examination reached. Please proceed to further working up the patient."
        return _prompt

    def process_investigations(self, query):
        requests = self.mapper.map_investigation_requests(query)

        self._log_requests(query, requests)

        _prompt = self.parser.parse_ix_requests(query, requests)

        self.ix_count += 1
        if self.ix_count >= self.ix_limit:
            _prompt += "\nLimit on ordering investigations reached. Please proceed to further working up the patient."

        return _prompt

    def process_imaging(self, query):
        requests = self.mapper.map_imaging_requests(query)

        self._log_requests(query, requests)

        _prompt = self.parser.parse_img_requests(query, requests)

        self.img_count += 1
        if self.img_count >= self.img_limit:
            _prompt += "\nLimit on ordering imaging reached. Please proceed to further working up the patient."

        return _prompt

    def get_examination_stats(self):

        matched_keys = set()
        for request_item in self.request_log:
            action = request_item["action"]
            for matched_request in request_item["matched"]:
                request_key = matched_request.get("key", "")
                matched_keys.add(f"{action}:{request_key}")

        unmatched_request_keys = set()
        for request_item in self.request_log:
            action = request_item["action"]
            for unmatched_request in request_item["unmatched"]:
                request_key = unmatched_request.get("key", "")
                unmatched_request_keys.add(f"{action}:{request_key}")

        unmatched_case_keys = set(self.clincase.keys()) - matched_keys

        return dict(
            action_count=self.action_count,
            hx_count=self.hx_count,
            phys_count=self.phys_count,
            ix_count=self.ix_count,
            img_count=self.img_count,
            action_limit_reached=self.action_count >= self.action_limit,
            hx_reached=self.hx_count >= self.hx_limit,
            phys_reached=self.phys_count >= self.phys_limit,
            ix_reached=self.ix_count >= self.ix_limit,
            img_reached=self.img_count >= self.img_limit,
            request_log=self.request_log,
            matched_keys=matched_keys,
            unmatched_case_keys=unmatched_case_keys,
            unmatched_request_keys=unmatched_request_keys,
            provisional_diagnosis=self.diagnosis_provisional,
            final_diagnosis=self.diagnosis_final,
        )


class Examination:

    def __init__(
        self,
        agent_model: BaseChatModel,
        clincase: ClinicalCase,
        examiner_model: BaseChatModel,
        examiner_kwargs={},
        turn_limit=20,
        logger=None,
    ):
        self.trace = []

        self.clincase = clincase
        self.logger = logger or _logger
        self.trace.append(self.clincase.full_information)
        self.logger.debug(f"\nFull Clinical Information:\n{clincase.full_information}")

        # LLM model being tested
        self.agent_model = agent_model
        self.agent_token_usage = 0
        self.agent_messages = [SystemMessage(ASSISTANT_BASE_PROMPT)]

        # Examiner model for information processing
        self.examiner = Examiner(
            clincase, examiner_model, logger=self.logger, **examiner_kwargs
        )

        self.stem = f"Clinical case stem: {self.clincase.demographics.prompt} presenting with {self.clincase.history.chief_complaint.lower()}.\n{self.clincase.physical.vitals.prompt}\nPlease review and diagnose the patient."
        self.agent_messages.append(HumanMessage(self.stem))

        self.examination_limit = turn_limit
        self.action_count = 0

        self.trace.append(self.stem)

        self.retry = 0

    def diagnose_with_full_information(self):

        full_information_stem = [
            SystemMessage(ASSISTANT_FULL_INFO_BASE_PROMPT),
            HumanMessage(self.clincase.full_information_no_ddx),
        ]

        invoke_success = False
        while not invoke_success:
            invoke_success, agent_response = self.invoke_agent(full_information_stem)

        self.logger.debug(f"\nDiagnosis with full information:\n{agent_response.query}")

        return agent_response

    def conduct_examination(self, test_full_info=True):

        if test_full_info:
            agent_response = self.diagnose_with_full_information()
            ddx_full_info = agent_response.query
        else:
            ddx_full_info = ""

        self.logger.debug(f"\nClinical Stem: {self.stem}")

        for _ in range(self.examination_limit):
            examiner_response = ""

            invoke_success, agent_response = self.invoke_agent(self.agent_messages)

            if not invoke_success:
                self.trace.append(str(agent_response))
                self.agent_messages.append(AIMessage(str(agent_response)))
                examiner_response = ERROR_RETURN_MSG + str(agent_response)

            else:
                try:
                    agent_response, examiner_response = self.examiner.process_response(
                        agent_response
                    )
                except Exception as e:
                    self.logger.error("Unable to process response")
                    self.logger.exception(str(agent_response))
                    self.logger.exception(e)
                    if self.retry == RETRY_LIMIT:
                        raise ValueError(f"Unable to parse agent response. {e}")

                    # Otherwise, retry a bit
                    self.trace.append(str(agent_response))
                    self.agent_messages.append(AIMessage(str(agent_response)))
                    examiner_response = ERROR_RETURN_MSG + str(agent_response)
                    self.retry += 1
                    invoke_success = False

            if invoke_success:

                self.agent_messages.append(AIMessage(agent_response.action_trace))
                self.trace.append(agent_response.full_trace)
                self.logger.debug(f"\nAgent Response: {agent_response.full_trace}")

                if agent_response.action == "diagnosis_final":

                    stats = self.examiner.get_examination_stats()
                    stats["agent_token_usage"] = self.agent_token_usage
                    stats["full_info_diagnosis"] = ddx_full_info

                    return (self.trace, stats)

            self.agent_messages.append(HumanMessage(examiner_response))

            self.trace.append(examiner_response)
            self.logger.debug(f"\nAgent Response: {examiner_response}")

        raise TimeoutError(
            "Turn limit reached - Increase examination turn limit or decrease examiner turn limit"
        )

    def invoke_agent(self, messages):

        response = self.agent_model.invoke(messages)
        agent_response = remove_json_markdown_enhanced(response.content)

        invoke_success = False

        try:
            rj = rapidjson.loads(agent_response)
            agent_response: AgentResponse = AgentResponse.model_validate(rj)
            self.retry = 0
            invoke_success = True

        except Exception as e:
            self.logger.error("Unable to parse agent response")
            self.logger.exception(f"=========== LLM response: {str(response)}")
            self.logger.exception(e)

            if self.retry == RETRY_LIMIT:
                raise ValueError(f"Unable to parse agent response. {e}")
            self.retry += 1
            invoke_success = False

        if response.usage_metadata:
            self.agent_token_usage += response.usage_metadata.get("total_tokens")

        return invoke_success, agent_response
