import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Dict, Literal

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger as _logger

from vivabench.ontology.schema import (
    ClinicalCase,
    InvestigationResult,
    PhysicalExamination,
    Symptom,
)
from vivabench.utils import prettify, remove_json_markdown, smart_capitalize


class ActionParser(ABC):
    """Parser processes routed requests from a router, retrieves relevant information from a clinical case with
    matched / unmatched keys, and parses it into a response to the LLM agent. The determinstic version is more
    robotic but robust, and the LLM version is more human readable? but prone to leakage
    """

    token_usage: int = 0

    @abstractmethod
    def __init__(self, clincase: ClinicalCase, logger=None):
        pass

    @abstractmethod
    def parse_history_requests(self, query: str, requests: dict) -> str:
        pass

    @abstractmethod
    def parse_physical_requests(self, query: str, requests: dict) -> str:
        pass

    @abstractmethod
    def parse_ix_requests(self, query: str, requests: dict) -> str:
        pass

    @abstractmethod
    def parse_img_requests(self, query: str, requests: dict) -> str:
        pass


class DeterminsticParser(ActionParser):

    def __init__(self, clincase: ClinicalCase, logger=None):

        self.logger = logger or _logger

        self.clincase = clincase

        self.hx_processed_keys = set()
        self.phys_processed_keys = set()
        self.ix_processed_keys = set()
        self.img_processed_keys = set()
        self.partial_keys = set()

        self.general_info_shown = False

        self.hx_matched_keys = set()
        self.phys_matched_keys = set()
        self.ix_matched_keys = set()
        self.img_matched_keys = set()

        self.hx_unmatched_keys = set()
        self.phys_unmatched_keys = set()
        self.ix_unmatched_keys = set()
        self.img_unmatched_keys = set()

    def _group_key_by_symptom(self, request_keys):
        grouped = {}
        for item in request_keys:
            key = item["key"]
            addit = item.get("addit", [])
            if key not in grouped:
                grouped[key] = set()
            grouped[key].update(addit)
        return [{"key": k, "addit": list(v)} for k, v in grouped.items()]

    def parse_history_requests(self, query, requests):
        matched_symptoms = []
        prim_prompt = ""
        sec_prompt = ""

        for request in self._group_key_by_symptom(requests.get("matched", [])):
            request_key = request.get("key")
            if request_key in self.hx_processed_keys:
                continue

            if request_key and ":" in request_key:

                request_group, request_item = request_key.split(":")

                if request_group == "symptoms":

                    sx_overall_key = f"{request_key}:general"
                    if sx_overall_key not in self.hx_processed_keys:
                        matched_symptoms.append(request_item)
                        self.hx_processed_keys.add(sx_overall_key)

                    addit_keys = request.get("addit", [])

                    sx_char_keys = [
                        f"{request_key}:{addit_key}" for addit_key in addit_keys
                    ]

                    # Filter for unprocessed characteristic keys only
                    _addit_keys = []
                    for addit_key, sx_char_key in zip(addit_keys, sx_char_keys):
                        if sx_char_key not in self.hx_processed_keys:
                            _addit_keys.append(addit_key)
                    if symptom := self.clincase.history.symptoms.get(request_item):
                        prim_prompt += symptom.get_prompt(_addit_keys) + "\n"
                    else:
                        requests["unmatched"] = requests.get("unmatched", []) + [
                            request
                        ]

                    for sx_char_key in sx_char_keys:
                        self.hx_processed_keys.add(sx_char_key)

                elif request_group == "hopc_structured":
                    request_group, request_item = request_key.split(":")

                    if hopc := self.clincase.history.hopc_structured.get(request_item):
                        prim_prompt += f"{prettify(request_item)} - {prettify(hopc)}"
                    else:
                        requests["unmatched"] = requests.get("unmatched", []) + [
                            request
                        ]
                    self.hx_processed_keys.add(request_key)

                elif request_group == "social_history":
                    if self.clincase.history.social_history.get(request_item):
                        sec_prompt += (
                            self.clincase.history.social_history.prompt(request_item)
                            + "\n"
                        )
                    self.hx_processed_keys.add(request_key)

            elif request_key == "family_history":
                if attr_list := self.clincase.history.family_history_list:
                    sec_prompt += f"{prettify(request_key)}:\n"
                    sec_prompt += f"{attr_list}\n"
                    self.hx_processed_keys.add(request_key)
            elif request_key == "past_medical_history":
                if attr_list := self.clincase.history.pmh_list:
                    sec_prompt += f"{prettify(request_key)}:\n"
                    sec_prompt += f"{attr_list}\n"
                    self.hx_processed_keys.add(request_key)
            elif request_key == "alleriges":
                if attr_list := self.clincase.history.allergies_list:
                    sec_prompt += f"{prettify(request_key)}:\n"
                    sec_prompt += f"{attr_list}\n"
                    self.hx_processed_keys.add(request_key)
            elif request_key == "medication_history":
                if attr_list := self.clincase.history.medication_list:
                    sec_prompt += f"{prettify(request_key)}:\n"
                    sec_prompt += f"{attr_list}\n"
                    self.hx_processed_keys.add(request_key)
            else:
                self.logger.warning(
                    "Unable to process history request: " + str(request)
                )

        if unmatched_requests := requests.get("unmatched", []):
            unmatched_hx = []
            for request in unmatched_requests:
                if request_key := request.get("key"):

                    if request_key in self.hx_processed_keys:
                        continue
                    unmatched_hx.append(request_key.split(":")[-1])
                else:
                    self.logger.warning(
                        "Unable to process history request: " + str(request)
                    )

            if unmatched_hx:

                sec_prompt = "\nNegative: "

                sec_prompt += prettify(", ".join(unmatched_hx)) + "."

        if hx_prompt := prim_prompt + sec_prompt:

            if matched_symptoms:
                sx_prompt = f"The patient experiences {', '.join([prettify(sx).lower() for sx in matched_symptoms])}.\n"
            else:
                sx_prompt = f"The patient does not have any other mentioned symptoms.\n"

            _prompt = sx_prompt + hx_prompt
        else:
            _prompt = "No more information on patient history available.\n"
        return _prompt

    def parse_physical_requests(self, query, requests):
        physical_by_systems = defaultdict(list)

        _prompt = ""

        physical = self.clincase.physical

        if not self.general_info_shown:
            _prompt += physical.vitals.prompt
            general_keys = [
                f"general:{k}" for k in physical.systems.get("general", {}).keys()
            ]
            if general_keys:
                _prompt += "General:\n"
                _prompt += "\n".join(physical.get_prompt(k) for k in general_keys)
                self.phys_processed_keys.update(set(general_keys))
            self.general_info_shown = True

        for request in requests.get("matched", []):

            if request_key := request.get("key"):
                if request_key in self.phys_processed_keys:
                    continue

                request_system = request_key.split(":")[0]

                if request_prompt := physical.get_prompt(request_key):

                    physical_by_systems[request_system].append(request_prompt)

                else:
                    self.logger.warning(
                        "Unable to process physical request: " + str(request)
                    )

                self.phys_processed_keys.add(request_key)

            else:
                self.logger.warning(
                    "Unable to process physical request: " + str(request)
                )

        for request in requests.get("unmatched", []):
            if request_key := request.get("key"):
                if request_key in self.phys_processed_keys:
                    continue

                request_split = request_key.split(":")

                if len(request_split) == 2:

                    request_system = request_split[0]

                    if not physical_by_systems[request_system]:
                        # If all negative, throw a default negative
                        physical_by_systems[request_system] = physical.get_default(
                            request_system
                        )
                        self.phys_processed_keys.add(request_system)

                    else:
                        # Otherwise, append negatives
                        physical_by_systems[request_system].append(
                            physical.get_default(request_key)
                        )
                self.phys_processed_keys.add(request_key)
            else:
                self.logger.warning(
                    "Unable to process physical request: " + str(request)
                )

        partial_matches = set()
        for request in requests.get("partial", []):
            if request_key := request.get("key"):
                request_split = request_key.split(":")
                if len(request_split) == 2:
                    if request_key not in self.partial_keys:
                        request_system = request_split[0]
                        partial_matches.add(request_system)
                        self.partial_keys.add(request_key)
            else:
                self.logger.warning(
                    "Unable to process physical request: " + str(request)
                )

        for partial_system in partial_matches:
            if not physical_by_systems[partial_system]:
                physical_by_systems[partial_system] = [
                    "Specify what you are looking for"
                ]

        for k, v in physical_by_systems.items():
            _prompt += prettify(k) + ": "
            _prompt += " ".join(v) + "\n"

        if not _prompt:
            _prompt = "No more physical examination results available."

        return _prompt

    def parse_ix_requests(self, query, requests):
        _request_keys = []
        for request in requests.get("matched", []):
            if request_key := request.get("key"):
                if request_key in self.ix_processed_keys:
                    continue
                if request_key not in self.ix_processed_keys:
                    _request_keys.append(request_key)
                    self.ix_processed_keys.add(request_key)
            else:
                self.logger.warning(
                    "Unable to process investigation request: " + str(request)
                )

        for request in requests.get("unmatched", []):
            if request_key := request.get("key"):
                if request_key in self.ix_processed_keys:
                    continue
                if request_key not in self.ix_processed_keys:
                    _request_keys.append(request_key)
                    self.ix_processed_keys.add(request_key)
            else:
                self.logger.warning(
                    "Unable to process investigation request: " + str(request)
                )

        if _request_keys:
            _prompt = self.clincase.investigations.get_grouped_investigations(
                _request_keys
            )
        else:
            _prompt = "No further investigation results available"

        return _prompt

    def parse_img_requests(self, query, requests):

        _prompt = ""
        for request in requests.get("matched", []):
            if request_key := request.get("key"):
                if request_key in self.img_processed_keys:
                    continue
                if imaging := self.clincase.imaging.get(request_key):
                    _prompt += imaging.prompt
                else:
                    requests["unmatched"] = requests.get("unmatched", []) + [request]
                self.img_processed_keys.add(request_key)
            else:
                self.logger.warning(
                    "Unable to process imaging request: " + str(request)
                )

        for request in requests.get("unmatched", []):
            if request_key := request.get("key"):
                if request_key in self.img_processed_keys:
                    continue
                _prompt += f"{request_key} not available.\n"
                self.img_processed_keys.add(request_key)
            else:
                self.logger.warning(
                    "Unable to process imaging request: " + str(request)
                )

        if not _prompt:
            _prompt = "No further imaging results available"
        return _prompt


HX_PARSE_SYSTEM = """You are simulating a patient responding to a doctor's questions. When responding:
1. Answer ONLY what was specifically asked in the query
2. Use natural, conversational language with minimal filler words
3. For information explicitly provided in the patient data, use that exact information
4. Be descriptive of the symptom in first person as if you are the patient experiencing it
5. For information NOT provided but reasonably expected:
   - Provide plausible responses that align with the overall clinical picture and diagnosis
   - Create responses that would be typical for a patient with the condition described
   - Respond with average medical literacy
   - Never contradict existing information or the established diagnosis
6. For negative findings, clearly state, with statements such as "I don't think I am experiencing [subjective symptom]" or "I don't think I have [symptom]" or "I don't have [condition]". However, the terminology also needs to be patient-focused as well. For example, a patient will not say "I don't have third nerve palsy".
7. Keep responses focused and appropriately detailed
"""

PHYS_PARSE_SYSTEM = """You are providing physical examination findings in a mock clinical exam. A student will describe what physical examination they would like to perform on the patient, and what specific physical examination findings they are looking for. 
When responding:
1. Address ONLY the specific examination findings requested in the query
2. Use brief, concise medical sentences with appropriate terminology
3. Format as a clinical note with system-based headers, with one line per system.
4. For examination findings mentioned in the provided information, return those exact findings
5. For examinating findings that is provided NOT mentioned in the query but relevant to the diagnosis, rephrase with non-specific, observable findings that is consistent with the patient's condition. It should not be any physical signs that could be elicited. Avoid overly dramatic or obvious findings.
   - For example, if the patient has appendicitis, with positive Rovsing's sign and rebound tenderness, and the student requests to perform an abdominal examination, but didn't specify to look for either Rovsing's sign or rebound tenderness, return "abdomen tender on palpation" 
6. For examination findings that is requested in the query but NOT in the provided examination findings:
   - If those examination findings are likely to be normal, provide appropriate negative findings (e.g. heart sounds dual)
   - If it is a specific sign that is negative for the diagnosis, cite negative (e.g. "Rovsing's sign negative")
   - If you are unsure if the requested examination finding will be positive in the patient or not, attribute to difficulties examining the patient (e.g. Unable to examine patient's reflexes)
7. Omit unnecessary details or explanations
8. Use standard medical abbreviations where appropriate

Remember: Be concise and directly address only what was asked. Your response should resemble the brief, focused documentation style used in clinical notes.
"""


class LLMParser(DeterminsticParser):
    """This is mostly used to 'humanize' the response LOL"""

    def __init__(self, clincase: ClinicalCase, model: BaseChatModel, logger=None):
        super().__init__(clincase=clincase, logger=logger)

        self.logger = logger or _logger

        self.clincase = clincase
        self._parser = DeterminsticParser(clincase)
        self.model = model

        self.hx_processed_attrs = {}

    def parse_history_requests(self, query, requests: Dict[str, Any]):
        """
        Parse history requests from a structured clinical case.

        Args:
            query (str): The original query from the doctor
            requests (dict): Dictionary containing matched and unmatched requests
            history (dict): The patient's history data

        Returns:
            str: Parsed response with history information
        """

        history = self.clincase.history
        req_prompt = "query: {query}\ninfo: {ans}"

        positive_qa_pairs = []
        negative_qa_pairs = []
        # Process matched requests
        for request in requests.get("matched", []):
            if request_key := request.get("key"):
                request_query = request.get("query", "")
                additional_attrs = request.get("addit", [])

                if request_key and ":" in request_key:
                    request_group, request_item = request_key.split(":")

                    # Handle symptoms
                    if request_group == "symptoms":
                        # For symptoms, we need to check if all requested attributes have been processed
                        # Initialize tracking for this symptom if it doesn't exist
                        symptom_key = f"{request_group}:{request_item}"

                        # Get or initialize the set of processed attributes for this symptom
                        if symptom_key not in self.hx_processed_attrs:
                            self.hx_processed_attrs[symptom_key] = set()

                        # Check if we've already processed all the requested attributes
                        requested_attrs_set = (
                            set(additional_attrs)
                            if additional_attrs
                            else set(["present"])
                        )
                        already_processed_attrs = requested_attrs_set.issubset(
                            self.hx_processed_attrs[symptom_key]
                        )

                        # Skip if we've already processed all requested attributes
                        if (
                            already_processed_attrs
                            and symptom_key in self.hx_matched_keys
                        ):
                            continue

                        symptom: Symptom = history.get(request_group, {}).get(
                            request_item
                        )

                        if symptom:
                            # Handle positive symptom
                            if symptom.present:
                                symptom_info = f"Positive: {symptom.name}"

                                if additional_attrs:
                                    symptom_info += "\n" + symptom.get_bullet(
                                        additional_attrs
                                    )
                                positive_qa_pairs.append(
                                    req_prompt.format(
                                        query=request_query, ans=symptom_info
                                    )
                                )

                                # Update processed attributes
                                if additional_attrs:
                                    self.hx_processed_attrs[symptom_key].update(
                                        additional_attrs
                                    )
                                else:
                                    self.hx_processed_attrs[symptom_key].add("present")
                            else:
                                # Handle relative negative symptom
                                negative_qa_pairs.append(
                                    req_prompt.format(
                                        query=request_query,
                                        ans=f"Negative: {request_item.replace('_', ' ')}",
                                    )
                                )
                                self.hx_processed_attrs[symptom_key].add("present")
                        else:
                            self.logger.warning(
                                "Symptom key not present: " + request_key
                            )
                            negative_qa_pairs.append(
                                req_prompt.format(
                                    query=request_query,
                                    ans=f"Negative: {request_item.replace('_', ' ')}",
                                )
                            )
                            self.hx_processed_attrs[symptom_key].add("present")

                        # Mark this symptom as matched
                        self.hx_matched_keys.add(symptom_key)
                    else:
                        # For non-symptom items, use the original logic
                        if request_key in self.hx_processed_keys:
                            continue

                        # Handle non-symptom items
                        hx_item = history.get(request_group, {}).get(request_item, {})
                        if hx_item:
                            if hasattr(hx_item, "bullet"):
                                ans = hx_item.bullet()
                            else:
                                ans = hx_item
                            positive_qa_pairs.append(
                                req_prompt.format(query=request_query, ans=ans)
                            )
                        else:
                            self.logger.warning(f"Key error: {request_key}")
                        self.hx_processed_keys.add(request_key)

                # Handle special history categories
                elif request_key in [
                    "family_history",
                    "past_medical_history",
                    "allergies",
                    "medication_history",
                ]:
                    if request_key in self.hx_processed_keys:
                        continue

                    attr_list_map = {
                        "family_history": "family_history_list",
                        "past_medical_history": "pmh_list",
                        "allergies": "allergies_list",
                        "medication_history": "medication_list",
                    }

                    attr_list_name = attr_list_map.get(request_key)
                    if attr_list_name and hasattr(history, attr_list_name):
                        attr_list = getattr(history, attr_list_name)
                        if attr_list:
                            positive_qa_pairs.append(
                                req_prompt.format(query=request_query, ans=attr_list)
                            )
                    self.hx_processed_keys.add(request_key)

                self.hx_matched_keys.add(request_key)
            else:
                self.logger.warning(f"Unable to process request: {request}")

        # Process unmatched requests
        for request in requests.get("unmatched", []):
            if request_key := request.get("key"):
                if request_key in self.hx_processed_keys:
                    continue
                request_query = request.get("query", "")

                if request_key and request_key not in self.hx_processed_keys:
                    if ":" in request_key:
                        request_group, request_item = request_key.split(":", 1)

                        negative_qa_pairs.append(
                            req_prompt.format(
                                query=request_query,
                                ans=f"Negative: {request_item.replace('_', ' ').replace(':', ' ')}",
                            )
                        )
                    else:
                        # Handle unmatched items without a specific key format
                        negative_qa_pairs.append(
                            req_prompt.format(
                                query=request_query, ans=f"No information available"
                            )
                        )

                self.hx_processed_keys.add(request_key)
                self.hx_unmatched_keys.add(request_key)
            else:
                self.logger.warning(f"Unable to process request: {request}")

        # LLM parsing to make it sound human
        if positive_qa_pairs or negative_qa_pairs:

            info = "\n".join(positive_qa_pairs) + "\n" + "\n".join(negative_qa_pairs)
            self.logger.debug(info)
            parse_prompt = f"Chief Complaint: {self.clincase.history.chief_complaint}\nDoctor query: {query}\nRelevant info:{info}"

            return self.model.invoke(
                [SystemMessage(HX_PARSE_SYSTEM), HumanMessage(parse_prompt)]
            ).content
        else:
            return "No more information on patient history available.\n"

    def parse_physical_requests(self, query, requests: Dict[str, Any]):
        _prompt = super().parse_physical_requests(query, requests)

        self.logger.debug(_prompt)
        parse_prompt = f"Chief Complaint: {self.clincase.history.chief_complaint}\nDoctor query: {query}\Examination Findings:{_prompt}"

        return self.model.invoke(
            [SystemMessage(PHYS_PARSE_SYSTEM), HumanMessage(parse_prompt)]
        ).content
