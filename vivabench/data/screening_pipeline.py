# Outdated script to screen for appropriate cases from MedQA. Included for reference

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock

import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tqdm import tqdm

from vivabench.generator import MedQACaseGenerator
from vivabench.ontology.schema import ClinicalCase, PhysicalExamination
from vivabench.utils import remove_json_markdown

ANTHROPIC_API = ""

OPENAI_API = ""

DEEPSEEK_API = ""

SCREENING_BASE = """You are a medical curriculum designer. Your job is to help me select and retrofit USMLE practice questions into clinical vignettes that could be used to test medical students in their diagnostic capacity in a Viva Voce examination. I will provide you with a question-answer pair, along with a provisional diagnosis of the disease from one of your colleagues. I want you to select the usable cases and filter out ones that might not be the most appropriate. 

Criteria:
1. This test is to test the diagnostic capacity of new medical doctors, and should focus on the diagnostic puzzle. As such, the clinical case needs to be a diagnosis for a new disease, and not a new complication to disease. however, it is OK for the disease to be secondary to pre-disposing factors (in fact actively encouraged.)
e.g. acceptable: alcohol-induced pancreatitis secondary to chronic alcoholism
not acceptable: Hypokalemia due to excessive insulin therap
2. Optimally, this would be a patient that one would see in an emergency setting, or in a primary care setting. If the patient in the vignette died, it's not useful
3. It should be a relatively difficult but verifiable diagnosis, in that all the relevant information required to diagnose the patient and make it distinct from other differential diagnoses is within the input. Bonus points if this is an important diagnosis, to which if it would be very bad if it was missed.
4. The diagnosis provided from your colleague should align well with the vignette. If the diagnosis was uncertain, or if you don't agree with the diagnosis, do not include.

I want you to return in json format, specifically {"reasoning": str , "usability", bool, "diagnosis": str}, where "diagnosis" is a single string from the input diagnosis with ICD-10 code
Example input: "A 64 year old man with upper abdominal pain. ECG showed ST-elevation"
Example output: {"reasoning": "This is a good case because it is an unusual presentation of STEMI, and it is high-stakes", "usability": true, "diagnosis": "I21.3 ST elevation (STEMI) myocardial infarction"}"""

SCREENING_TEMPLATE = """Practice Exam Question: {question}. Practice Exam Answer: {answer}. Your colleague's diagnosis: {diagnosis}. Is this case fit for the examination? I want you to return in json format, specifically "reasoning": str , "usability", bool, "diagnosis": str, where "diagnosis" is a single string from the input diagnosis with ICD-10 code"""

MEDQA_PROMPT = """{question} Options: {options}. Return single letter answer only"""
DDX_PROMPT = """Given this following USMLE question: {question} Options: {options} Answer: {answer}, What is the diagnosis? Return in ICD-10 code + phrase"""

MEDQA_PROMPT = """{question} Options: {options}. Return single letter answer only"""
DDX_PROMPT_FULL = """Given this following USMLE question: {question} Options: {options} Answer: {answer}, What is the diagnosis? Return in ICD-10 code + phrase"""
DDX_PROMPT_SHORT = """Given the clinical vignette within this USMLE question: {question}What is the diagnosis? Return single phrase only"""
DDX_COMPARISON = """These are the answers from two students for an examination: {a1}, {a2}. This is the answer: {answer}. Are either students correct? If they are similar in definition, it can be considered correct. However, if they have an entirely wrong diagnosis, then they are considered incorrect. Return in json format "student_1": bool, "student_2": bool """

# Number of workers
NUM_WORKERS = 4

medqa = pd.read_json(path_or_buf="medqa_train.jsonl", lines=True)
medqa = medqa[7600:]

# Rate limiting parameters
RATE_LIMIT = 10  # requests per second
BUCKET_CAPACITY = 10  # maximum burst capacity


class RateLimiter:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            time_passed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + time_passed * self.rate)
            self.last_refill = now

            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                time.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1


rate_limiter = RateLimiter(RATE_LIMIT, BUCKET_CAPACITY)


def create_api_clients():
    return {
        "gpt": init_chat_model(
            "openai:gpt-4.1-mini", temperature=0, api_key=OPENAI_API
        ),
        "claude": init_chat_model(
            "deepseek:deepseek-chat", temperature=0, api_key=DEEPSEEK_API
        ),
        "gpt41": init_chat_model("openai:gpt-4.1", temperature=0, api_key=OPENAI_API),
    }


async def rate_limited_invoke(model, prompt):
    rate_limiter.acquire()
    return await model.ainvoke(prompt)


async def process_row(row, api_clients):
    try:
        gpt_41_mini = api_clients["gpt"]
        # claude = api_clients['claude']
        gpt_41 = api_clients["gpt41"]

        consider_use_case = False
        case_passed_screening = False
        ddx = None

        # First, check if either of the models get the question wrong
        input_prompt = MEDQA_PROMPT.format(
            question=row["question"], options=row["options"]
        )

        # Make concurrent API calls with rate limiting
        gpt_task = rate_limited_invoke(gpt_41_mini, input_prompt)
        # claude_task = rate_limited_invoke(claude, input_prompt)

        # Await both responses
        gpt_response = (await gpt_task).content
        # claude_response = (await claude_task).content

        if (
            gpt_response != row["answer_idx"]
        ):  # and claude_response != row['answer_idx']:
            consider_use_case = True

        if not consider_use_case:
            # Then, also check if the ddx is wrong
            input_prompt = DDX_PROMPT_SHORT.format(
                question=".".join(row["question"].split(".")[:-1])
            )

            # Run these API calls concurrently
            gpt_task = gpt_41_mini.ainvoke(input_prompt)
            # claude_task = claude.ainvoke(input_prompt)

            _ddx_prompt = DDX_PROMPT_FULL.format(
                question=row["question"], answer=row["answer"], options=row["options"]
            )
            ddx_task = gpt_41.ainvoke(_ddx_prompt)

            # Await all responses
            gpt_response = (await gpt_task).content
            # claude_response = (await claude_task).content
            ddx = (await ddx_task).content

            comparison_task = gpt_41.ainvoke(
                DDX_COMPARISON.format(a1=gpt_response, a2=gpt_response, answer=ddx)
            )
            comparison = (await comparison_task).content
            ans = json.loads(remove_json_markdown(comparison))

            if not ans["student_1"] and not ans["student_2"]:
                consider_use_case = True

        if consider_use_case:
            case_screening = SCREENING_TEMPLATE.format(
                question=row["question"], answer=row["answer"], diagnosis=ddx
            )
            messages = [SystemMessage(SCREENING_BASE), HumanMessage(case_screening)]

            screening_result = json.loads((await gpt_41_mini.ainvoke(messages)).content)

            if screening_result["usability"]:
                ddx = screening_result["diagnosis"]
                case_passed_screening = True

        if case_passed_screening:
            row_dict = row.to_dict()

            row_dict["gpt"] = gpt_response
            row_dict["claude"] = None
            row_dict["ddx"] = ddx
            row_dict["reasoning"] = screening_result["reasoning"]

            print("Collected case with diagnosis:", ddx)

            with open("medqa_output_0504_train.jsonl", "a") as f:
                f.write(json.dumps(row_dict) + "\n")

            return True
        return False

    except Exception as e:
        print(f"Error processing row: {e}")
        return False


async def process_batch(batch):
    api_clients = create_api_clients()
    tasks = [process_row(row, api_clients) for _, row in batch.iterrows()]
    return await asyncio.gather(*tasks)


async def process_batch(batch, api_clients):
    tasks = [process_row(row, api_clients) for _, row in batch.iterrows()]
    return await asyncio.gather(*tasks)


async def main():
    api_clients = create_api_clients()

    # Split the dataframe into batches
    batch_size = 100  # Adjust this based on your needs
    batches = [medqa[i : i + batch_size] for i in range(0, len(medqa), batch_size)]

    for batch in tqdm(batches):
        await process_batch(batch, api_clients)

    # Close API clients
    for client in api_clients.values():
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
