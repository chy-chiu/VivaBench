from vivabench.util import hfapi_decode, chatgpt_decode
from vivabench.util import str_to_msgs
from vivabench.ontology.concepts import ClinicalCase
from typing import Literal

LLM_RETRIES = 5

## TODO: Need to move all the prompts into the prompt section instead

class Examiner:
    def __init__(
        self, clinical_case: ClinicalCase, examiner_model, **llm_kwargs
    ) -> None:
        self.clinical_case = clinical_case

        # Will sort out these sort of configs later + refactor to separate section
        max_length = llm_kwargs.get("max_length", 256)
            
        if "gpt" in examiner_model:
            self.llm = lambda x: chatgpt_decode(x, examiner_model, max_length)
        elif "mistral" in examiner_model or "mixtral" in examiner_model:
            self.llm = lambda x: hfapi_decode(x, examiner_model, max_length)

        # TODO: Refactor this out
        self.ddx_prompt = open(
            "../vivabench/prompts/examiner_differentials.prompt", "r"
        ).read()
        self.bedside_prompt = open(
            "../vivabench/prompts/examiner_bedside.prompt", "r"
        ).read()
        self.ix_prompt = open(
            "../vivabench/prompts/examiner_investigations.prompt", "r"
        ).read()
        self.mx_prompt = open(
            "../vivabench/prompts/examiner_management.prompt", "r"
        ).read()

        self.f1_scores = []
        self.step: Literal[
            "bedside", "differentials", "investigations", "diagnosis", "management", "termination"
        ] = "bedside"

    def initial_prompt(self):

        prompt = self.clinical_case.hopc.brief_history
        prompt += "\nYou are currently reviewing this patient.\n[[QUESTION]] What information would you like to seek from your history and examination?"

        return prompt

    def format_examiner_prompt(self, agent_response, prompt_type):
        clinical_case = self.clinical_case

        if prompt_type == "ddx":
            if self.step == "differentials":
                ddx = clinical_case.primary_ddx
            elif self.step == "diagnosis":
                ddx = clinical_case.secondary_ddx

            prompt = self.ddx_prompt.format(
                agent_response=agent_response,
                differentials=ddx.prompt(),
            )
        elif prompt_type == "bedside":
            prompt = self.bedside_prompt.format(
                agent_response=agent_response,
                hopc_full=clinical_case.hopc.full_history,
                systems_review=clinical_case.systems.prompt(),
                physical_examination=clinical_case.examination.prompt(),
                pmh=clinical_case.pmh.prompt(),
            )
        elif prompt_type == "ix":
            available_ix = list(clinical_case.ix.keys())
            prompt = self.ix_prompt.format(
                hopc_full=clinical_case.hopc.full_history,
                agent_response=agent_response,
                investigations=available_ix,
            )
        elif prompt_type == "mx":
            prompt = self.mx_prompt.format(
                agent_response=agent_response, management=clinical_case.management
            )

        prompt_parsed = str_to_msgs(prompt)

        return prompt_parsed

    def reply_bedside(self, agent_response):
        prompt_parsed = self.format_examiner_prompt(
            agent_response, prompt_type="bedside"
        )
        response = self.llm(prompt_parsed)

        # TODO: Here if bedside is not sufficient we can do a second prompt, but let's assume it's good enough.
        response_satisfactory = True
        if response_satisfactory:
            response += "\n\n[[QUESTION]] With this information, what is your differential diagnosis for this patient?"
            self.step = "differentials"
        else:
            response += "\n\n[[QUESTION]] What else would you like to ask the patient?"

        return response

    @staticmethod
    def check_json_response(response):
        # Mixtral has weird backslash in the name. Will remove.
        response = response.replace("\\", "").strip("`").strip("json")
        if type(eval(response)) == dict:
            # Will need more checks re output later. but this will do for now
            return True, eval(response)
        else:
            return False, response

    @staticmethod
    def calculate_f1(response):
        tp = len(response["true_positive"])
        fp = len(response["false_positive"])
        fn = len(response["false_negative"])

        return tp / (tp + (fp + fn) * 0.5)

    def reply_ddx(self, agent_response):
        retries = 0

        prompt_parsed = self.format_examiner_prompt(
            agent_response, prompt_type="ddx"
        )
        response = self.llm(prompt_parsed)

        success, response = Examiner.check_json_response(response)

        while retries < LLM_RETRIES and not success:
            response = self.llm(prompt_parsed)
            success, response = Examiner.check_json_response(response)
            retries += 1

        if not success:
            return None  # sort out later will need to raise some issue

        # TODO: Here, we can try again if the score is low. to do it later
        f1 = Examiner.calculate_f1(response)
        self.f1_scores.append(f1)

        if self.step == "differentials":
            response = (
                "Thank you. \n[[QUESTION]] What investigations would you order for this patient?"
            )

            self.step = "investigations"

        elif self.step == "diagnosis":
            response = f"Thank you. The correct diagnosis for this patient is:\n{self.clinical_case.ddx.prompt()}\n[[QUESTION]] How would you manage this patient?"

            self.step = "management"

        return response

    def reply_ix(self, agent_response):
        prompt_parsed = self.format_examiner_prompt(agent_response, prompt_type="ix")

        # TODO: Need to make this more robust. Will fix it later
        requested_ix = eval(self.llm(prompt_parsed))
        available_ix = list(self.clinical_case.ix.keys())

        ordered_ix = set(requested_ix).intersection(set(available_ix))

        response = "Here are the available investigations:\n"
        response += "\n".join(
            [self.clinical_case.ix[ix].prompt() for ix in list(ordered_ix)]
        )
        response += "[[QUESTION]] What is your current diagnosis?"

        self.step = "diagnosis"

        return response

    def reply_mx(self, agent_response):
        prompt_parsed = self.format_examiner_prompt(agent_response, prompt_type="mx")

        response = self.llm(prompt_parsed)

        # TODO: Similar to ddx, will need more checks to ensure robustness. But it's good enough for now
        success, response = Examiner.check_json_response(response)

        # TODO: Here, we can try again if the score is low. to do it later
        f1 = Examiner.calculate_f1(response)
        self.f1_scores.append(f1)

        response = "Examination finished. Thank you"

        self.step = "termination"

        return response

    def reply(self, agent_response):
        """Depending on step in the viva, pick examiner response accordingly."""
        if self.step == "bedside":
            examiner_response = self.reply_bedside(agent_response)
        elif self.step == "investigations":
            examiner_response = self.reply_ix(agent_response)
        elif self.step == "management":
            examiner_response = self.reply_mx(agent_response)
        elif self.step == "differentials" or self.step == "diagnosis":
            examiner_response = self.reply_ddx(agent_response)
        elif self.step == "termination":
            raise ValueError("Examination terminated")

        return examiner_response
