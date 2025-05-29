# Old script to screen for appropriate cases from PubMed. Included for reference

import asyncio
import json
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from vivabench.utils import remove_json_markdown


class CaseReportFilterPipeline:
    def __init__(
        self,
        model_name: str = "openai:gpt-4.1-mini",
        temperature: float = 0.5,
        api_key: str = None,
        total_limit: int = 1000,
        group_limit: int = 150,
        min_score: int = 9,
        batch_size: int = 1000,
        max_concurrent: int = 10,
    ):
        """
        Initialize the case report filtering pipeline.

        Args:
            model_name: The LLM model to use
            temperature: Temperature setting for the model
            api_key: API key for the model service
            total_limit: Maximum total cases to collect
            group_limit: Maximum cases per specialty group
            min_score: Minimum score (1-10) to accept a case
            batch_size: Number of cases to process in each batch
            max_concurrent: Maximum number of concurrent API calls
        """
        self.model = init_chat_model(
            model_name, temperature=temperature, api_key=api_key
        )
        self.total_limit = total_limit
        self.group_limit = group_limit
        self.min_score = min_score
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent

        # Track collected cases
        self.collected_cases = []
        self.group_counts = defaultdict(int)
        self.processed_ids = set()

        # Set up logging
        logger.add("case_filter_pipeline.log", rotation="100 MB")

        # Load prompts
        self.title_prompt = self._load_title_prompt()
        self.fulltext_prompt = self._load_fulltext_prompt()

    def _load_title_prompt(self) -> str:
        """Load the title analysis prompt"""
        return """
       You are a specialized medical case report evaluator with expertise in identifying diagnostically rich and educationally valuable clinical cases. Your task is to analyze the TITLE of a medical case report and determine if it meets our criteria for further human review.

EVALUATION CRITERIA:
1. The case should involve a human patient (not animal medicine)
2. The case should focus on the diagnostic journey rather than treatment specifics
3. The case should be diagnosable by general practitioners or emergency physicians (not requiring subspecialist expertise)
4. Cases involving missed or delayed diagnoses with clinical consequences are valuable
5. The case should NOT involve extremely rare diseases or require highly specialized testing
6. The case should NOT primarily focus on management/treatment

IMPORTANT NOTE ON PEDIATRIC AND ONCOLOGY CASES:
- Pediatric cases ARE valuable if they involve diagnostic challenges, missed diagnoses, or atypical presentations that would be educational for general practitioners
- Oncology cases ARE valuable if they involve cancer masquerading as something else, missed diagnoses, or atypical presentations
- AVOID pediatric cases focusing on rare congenital disorders or highly specialized pediatric conditions
- AVOID oncology cases focusing on rare cancer subtypes, molecular characterization, or specialized oncology treatments

STRICT SCORING GUIDELINES:
- Score 10: Reserved ONLY for exceptional titles that clearly indicate a diagnostically rich case with substantial educational value for general practice. Must explicitly suggest diagnostic challenges, misdiagnoses, or atypical presentations of conditions commonly encountered in general practice.

- Score 9: Excellent titles that strongly indicate diagnostic content with clear educational value, but may not be as explicitly focused on diagnostic challenges as a 10.

- Score 7-8: Good titles that suggest diagnostic content but may have minor limitations or less clarity about the diagnostic focus.

- Score 5-6: Average titles that could be diagnostic in nature but lack clear indicators or may have some treatment focus.

- Score 1-4: Poor titles that clearly indicate animal cases, ultra-specialized content, primarily treatment focus, or extremely rare conditions.

NEGATIVE INDICATORS IN TITLES (SCORE REDUCERS):
- Animal subjects (e.g., "in a dog," "in mice") [automatic 1-3 score]
- Highly specialized genetic or molecular focus (e.g., "Novel Intronic Variant," "Gene Expression") [reduce score by 2 points]
- Extremely rare diseases or syndromes [reduce score by 2 points]
- Heavy focus on treatment modalities (e.g., "after Stereotactic Radiation," "Response to Therapy") [reduce score by 2 points]
- Highly subspecialized contexts (e.g., "Opportunities for Precision Radiation") [reduce score by 2 points]
- Excessive technical jargon suggesting subspecialist audience [reduce score by 1-2 points]
- Rare pediatric congenital disorders [reduce score by 2 points]
- Rare cancer subtypes or molecular characterization [reduce score by 2 points]

POSITIVE INDICATORS IN TITLES (SCORE ENHANCERS):
- Diagnostic challenges (e.g., "Misdiagnosed as," "Masked by," "Complicated with") [add 2-3 points]
- Common conditions with atypical presentations [add 2 points]
- Diagnostic reasoning elements (e.g., "Inadequate Physical Examination," "Narrow Focus Thinking") [add 2-3 points]
- Presentations that could be encountered in general or emergency practice [add 1-2 points]
- Uncommon but recognizable presentations of known conditions [add 1 point]
- Mentions of diagnostic processes rather than treatments [add 1 point]
- Cancer masquerading as another condition [add 2 points]
- Missed pediatric diagnoses with educational value [add 2 points]

IMPORTANT: You must respond in valid JSON format with the following fields:
- score: A number from 1-10 representing your evaluation (be very selective with 9-10 scores)
- explanation: Brief explanation of your rating, including which positive and negative indicators influenced your score
- specialty_group: Classify into one of these groups: "Cardiovascular & Metabolic", "Respiratory", "Gastrointestinal", "Musculoskeletal & Pain", "Neurological / Psychiatric", "Infectious Disease & Immunology", "Endocrine & Reproductive", "Pediatric", "Other"
- is_human: Boolean indicating if this is definitely a human case (false for animal cases)
        """

    def _load_fulltext_prompt(self) -> str:
        """Load the full text analysis prompt"""
        return """
        You are a specialized medical case report evaluator with expertise in identifying diagnostically rich and educationally valuable clinical cases. Your task is to analyze the FULL TEXT of a medical case report and determine if it meets our criteria for further human review.

EVALUATION CRITERIA:
1. The case must involve a human patient (not animal medicine)
2. The case should focus primarily on the diagnostic journey rather than treatment specifics
3. The case should be diagnosable by general practitioners or emergency physicians
4. The case should contain rich clinical information including:
   - Detailed history and physical examination
   - Relevant laboratory and imaging findings with specific values
   - Clear diagnostic reasoning process
5. The case should represent a first presentation or a diagnostic challenge
6. Cases involving missed or delayed diagnoses with clinical consequences are valuable
7. The case should NOT involve extremely rare diseases or require highly specialized testing
8. The case should NOT primarily focus on management/treatment

IMPORTANT NOTE ON PEDIATRIC AND ONCOLOGY CASES:
- Pediatric cases ARE valuable if they involve diagnostic challenges, missed diagnoses, or atypical presentations that would be educational for general practitioners
- Oncology cases ARE valuable if they involve cancer masquerading as something else, missed diagnoses, or atypical presentations
- AVOID pediatric cases focusing on rare congenital disorders or highly specialized pediatric conditions
- AVOID oncology cases focusing on rare cancer subtypes, molecular characterization, or specialized oncology treatments

STRICT SCORING GUIDELINES:
- Score 10: Reserved ONLY for truly exceptional cases that meet ALL of these criteria:
  * Rich, detailed history and physical examination
  * Multiple specific laboratory values with units
  * Clear imaging findings relevant to diagnosis
  * Well-documented diagnostic reasoning process
  * Represents a diagnostic challenge or missed diagnosis with clear learning points
  * Condition that could be encountered in general practice
  * Minimal focus on treatment/management
  * Contains explicit discussion of differential diagnoses

- Score 9: Excellent cases that meet nearly all criteria for a 10, but may be slightly less detailed in one area or have minor limitations.

- Score 7-8: Good cases with substantial diagnostic information but have clear limitations in 2-3 areas.

- Score 5-6: Average cases with some diagnostic information but significant limitations in multiple areas.

- Score 1-4: Poor cases that fail to meet multiple criteria or focus primarily on excluded topics.

REFERENCE EXAMPLE OF A 10/10 CASE:
A case report of recurrent acute pancreatitis with detailed clinical history, laboratory data with specific values, multiple imaging modalities (CT, ultrasound, MRCP, ERCP), and a clear diagnostic challenge involving a duodenal ulcer scar causing ampullary stricture and distortion leading to pancreatitis. The diagnostic journey is well documented, including initial negative findings and eventual successful diagnosis. The case has rich diagnostic information, clear reasoning, and high educational value.

CONTENT STRUCTURE ASSESSMENT:
Evaluate the presence and quality of these key sections:
- Patient history (demographics, presenting complaints, timeline)
- Physical examination findings
- Laboratory investigations with specific values
- Imaging studies with findings
- Diagnostic reasoning process
- Temporal sequence of the diagnostic journey

NEGATIVE INDICATORS (SCORE REDUCERS):
- Animal subjects [automatic 1-3 score]
- Post-procedure or post-operative complications as the primary focus [reduce score by 2 points]
- Highly specialized genetic or molecular focus [reduce score by 2 points]
- Extremely rare diseases requiring subspecialist knowledge [reduce score by 2 points]
- Heavy focus on treatment modalities rather than diagnosis [reduce score by 2-3 points]
- Lack of detailed clinical information [reduce score by 2-3 points]
- Absence of diagnostic reasoning elements [reduce score by 2 points]
- Absence of laboratory values with units [reduce score by 1-2 points]
- Absence of imaging findings [reduce score by 1-2 points if relevant to the case]
- Rare pediatric congenital disorders [reduce score by 2 points]
- Rare cancer subtypes or molecular characterization [reduce score by 2 points]
- Highly specialized oncology treatments [reduce score by 2 points]

POSITIVE INDICATORS (SCORE ENHANCERS):
- Rich history and physical examination details [add 1-2 points]
- Multiple relevant laboratory values with units [add 1-2 points]
- Clear imaging findings related to diagnosis [add 1-2 points]
- Explicit diagnostic challenges or dilemmas [add 2 points]
- Missed or delayed diagnoses with learning points [add 2-3 points]
- Common conditions with atypical presentations [add 2 points]
- Logical clinical flow and temporal sequence [add 1 point]
- Clear educational value for general practitioners [add 1-2 points]
- Explicit discussion of differential diagnoses [add 1-2 points]
- Cancer masquerading as another condition [add 2 points]
- Missed pediatric diagnoses with educational value [add 2 points]
- Atypical presentation of common pediatric conditions [add 2 points]

IMPORTANT: You must respond in valid JSON format with the following fields:
- score: A number from 1-10 representing your evaluation (be very selective with 9-10 scores)
- explanation: Brief explanation of your rating, including which positive and negative indicators influenced your score
- specialty_group: Classify into one of these groups: "Cardiovascular & Metabolic", "Respiratory", "Gastrointestinal", "Musculoskeletal & Pain", "Neurological / Psychiatric", "Infectious Disease & Immunology", "Endocrine & Reproductive", "Pediatric", "Other"
- is_human: Boolean indicating if this is definitely a human case (false for animal cases)
- diagnosis: The primary diagnosis in the case
- differentials: List of differential diagnoses discussed in the case report
- key_learning_points: Brief list of key diagnostic learning points from this case
"""

    async def async_model_invoke(
        self, messages: List[SystemMessage | HumanMessage]
    ) -> Tuple[AIMessage, int]:
        """
        Asynchronously invoke the language model.

        Args:
            messages: List of messages to send to the model

        Returns:
            Tuple of (model response, token count)
        """
        try:
            response: AIMessage = await self.model.ainvoke(messages)
            return response, response.usage_metadata["total_tokens"]
        except Exception as e:
            logger.error(f"Error invoking model: {e}")
            # Return a default error response
            return (
                AIMessage(
                    content=json.dumps(
                        {
                            "score": 0,
                            "explanation": f"Error: {str(e)}",
                            "specialty_group": "Other",
                            "is_human": False,
                        }
                    )
                ),
                0,
            )

    async def process_title(self, case_id: str, title: str) -> Dict:
        """
        Process a case report title.

        Args:
            case_id: Unique identifier for the case
            title: The title of the case report

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        system_prompt = self.title_prompt
        human_prompt = "TITLE TO EVALUATE:\n{title}".format(title=title)
        artifact = {}
        try:
            response, token_count = await self.async_model_invoke(
                [SystemMessage(system_prompt), HumanMessage(human_prompt)]
            )
            artifact["response"] = response.content
            # Parse JSON response
            result = json.loads(remove_json_markdown(response.content))

            # Add metadata
            result["PMID"] = case_id
            result["title"] = title
            result["processing_time"] = time.time() - start_time
            result["token_count"] = token_count

            logger.debug(
                f"Processed title for case {case_id}: Score {result.get('score', 0)}"
            )
            return result

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from response for case {case_id}")
            logger.debug(artifact["response"])
            return {
                "PMID": case_id,
                "title": title,
                "score": 0,
                "explanation": "Error: Failed to parse response",
                "specialty_group": "Other",
                "is_human": False,
                "processing_time": time.time() - start_time,
                "token_count": 0,
            }

    async def process_fulltext(self, case_id: str, title: str, text: str) -> Dict:
        """
        Process the full text of a case report.

        Args:
            case_id: Unique identifier for the case
            title: The title of the case report
            text: The full text of the case report

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        system_prompt = self.fulltext_prompt
        human_prompt = "CASE REPORT TO EVALUATE:\n{text}".format(text=text)
        artifact = {}

        try:
            response, token_count = await self.async_model_invoke(
                [SystemMessage(system_prompt), HumanMessage(human_prompt)]
            )
            artifact["response"] = response.content

            # Parse JSON response
            result = json.loads(remove_json_markdown(response.content))

            # Add metadata
            result["PMID"] = case_id
            result["title"] = title
            result["text"] = text
            result["processing_time"] = time.time() - start_time
            result["token_count"] = token_count

            logger.info(
                f"Processed fulltext for case {case_id}: Score {result.get('score', 0)}"
            )
            return result

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from response for case {case_id}")
            logger.debug(artifact["response"])

            return {
                "PMID": case_id,
                "title": title,
                "text": text,
                "score": 0,
                "explanation": "Error: Failed to parse response",
                "specialty_group": "Other",
                "is_human": False,
                "diagnosis": "Unknown",
                "differentials": [],
                "processing_time": time.time() - start_time,
                "token_count": 0,
            }

    async def process_batch_titles(self, batch: List[Dict]) -> List[Dict]:
        """
        Process a batch of case report titles in parallel.

        Args:
            batch: List of dictionaries with case_id and title

        Returns:
            List of processing results
        """
        tasks = []
        for case in batch:
            if case["PMID"] in self.processed_ids:
                continue

            tasks.append(self.process_title(case["PMID"], case["title"]))

        results = await tqdm_asyncio.gather(*tasks, desc="Processing titles")
        return results

    async def process_batch_fulltexts(self, batch: List[Dict]) -> List[Dict]:
        """
        Process a batch of case report full texts in parallel.

        Args:
            batch: List of dictionaries with case_id, title, and text

        Returns:
            List of processing results
        """
        tasks = []
        for case in batch:
            if case["PMID"] in self.processed_ids:
                continue

            tasks.append(
                self.process_fulltext(case["PMID"], case["title"], case["text"])
            )

        results = await tqdm_asyncio.gather(*tasks, desc="Processing full texts")
        return results

    def should_process_case(self, specialty_group: str) -> bool:
        """
        Determine if we should process a case based on group limits.

        Args:
            specialty_group: The specialty group of the case

        Returns:
            Boolean indicating if we should process the case
        """
        # Check if we've reached the total limit
        if len(self.collected_cases) >= self.total_limit:
            return False

        # Check if we've reached the group limit
        if self.group_counts[specialty_group] >= self.group_limit:
            return False

        return True

    def add_case(self, case: Dict) -> bool:
        """
        Add a case to our collection if it meets criteria.

        Args:
            case: Case data dictionary

        Returns:
            Boolean indicating if the case was added
        """
        # Check if case meets minimum score
        if case.get("score", 0) < self.min_score:
            return False

        # Check if case is for a human
        if not case.get("is_human", True):
            logger.warning(f"Skipping non-human case: {case['PMID']}")
            return False

        # Check group limits
        specialty_group = case.get("specialty_group", "Other")
        if not self.should_process_case(specialty_group):
            logger.info(
                f"Skipping case due to limits: {case['PMID']} ({specialty_group})"
            )
            return False

        # Add the case
        self.collected_cases.append(case)
        self.processed_ids.add(case["PMID"])
        self.group_counts[specialty_group] += 1

        logger.success(
            f"Added case {case['PMID']} to collection (Group: {specialty_group}, Score: {case.get('score', 0)})"
        )
        return True

    async def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process all case reports in the dataset.

        Args:
            data: DataFrame with case_id, title, and text columns

        Returns:
            DataFrame with filtered and processed cases
        """
        logger.info(f"Starting processing of {len(data)} case reports")

        # Process in batches
        for i in range(0, len(data), self.batch_size):
            batch = data.iloc[i : i + self.batch_size].to_dict("records")
            logger.info(
                f"Processing batch {i//self.batch_size + 1}/{(len(data)-1)//self.batch_size + 1}"
            )

            # First pass: title filtering
            title_results = await self.process_batch_titles(batch)

            # Filter promising cases for full-text analysis
            promising_cases = []
            for result in title_results:
                # if result.get("score", 0) >= 7:
                #     logger.debug(result)
                if (
                    result.get("score", 0) >= 7
                    and result.get("is_human", True)
                    and self.should_process_case(result.get("specialty_group", "Other"))
                ):

                    # Find the full text for this case
                    case_id = result["PMID"]
                    case_data = next(
                        (item for item in batch if item["PMID"] == case_id), None
                    )

                    if case_data and "patient" in case_data:
                        promising_cases.append(
                            {
                                "PMID": case_id,
                                "title": result["title"],
                                "text": case_data["patient"],
                            }
                        )

            logger.info(
                f"Found {len(promising_cases)} promising cases for full-text analysis"
            )

            # Second pass: full-text analysis (in smaller concurrent batches)
            for j in range(0, len(promising_cases), self.max_concurrent):
                sub_batch = promising_cases[j : j + self.max_concurrent]
                fulltext_results = await self.process_batch_fulltexts(sub_batch)

                # Add high-scoring cases to our collection
                for result in fulltext_results:
                    self.add_case(result)

                # Check if we've reached our total limit
                if len(self.collected_cases) >= self.total_limit:
                    logger.info(f"Reached total limit of {self.total_limit} cases")
                    break

            # Check if we've reached our total limit
            if len(self.collected_cases) >= self.total_limit:
                break

            # Log progress
            logger.info(
                f"Current collection: {len(self.collected_cases)}/{self.total_limit} total cases"
            )
            for group, count in self.group_counts.items():
                logger.info(f"  - {group}: {count}/{self.group_limit} cases")

            # Convert collected cases to DataFrame
            pd.DataFrame(self.collected_cases).to_csv("_ckpt_df.csv", index=False)

        # Convert collected cases to DataFrame
        result_df = pd.DataFrame(self.collected_cases)

        # Ensure we have the required columns
        required_columns = [
            "title",
            "text",
            "diagnosis",
            "differentials",
            "score",
            "specialty_group",
        ]
        for col in required_columns:
            if col not in result_df.columns:
                result_df[col] = None

        # Format differentials as string if it's a list
        if "differentials" in result_df.columns:
            result_df["differentials"] = result_df["differentials"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else x
            )

        logger.success(f"Completed processing with {len(result_df)} cases collected")
        return result_df[
            ["title", "text", "diagnosis", "differentials", "score", "specialty_group"]
        ]

    def save_results(self, output_path: str = "filtered_cases.csv"):
        """
        Save the collected cases to a CSV file.

        Args:
            output_path: Path to save the CSV file
        """
        result_df = pd.DataFrame(self.collected_cases)

        # Format differentials as string if it's a list
        if "differentials" in result_df.columns:
            result_df["differentials"] = result_df["differentials"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else x
            )

        # Select and rename columns for the output format
        output_df = result_df[
            ["title", "text", "diagnosis", "differentials", "score", "specialty_group"]
        ]
        output_df = output_df.rename(
            columns={
                "score": "LLM Score for appropriateness",
                "differentials": "Other differentials discussed in the case report",
            }
        )

        output_df.to_csv(output_path, index=False)
        logger.success(f"Saved {len(output_df)} cases to {output_path}")


class ClinicalCaseAnalysisPipeline:
    def __init__(
        self,
        model_name: str = "openai:gpt-4.1-mini",
        temperature: float = 0.2,
        api_key: str = None,
        max_workers: int = 4,
        batch_size: int = 10,
        max_retries: int = 5,
    ):
        """
        Initialize the clinical case analysis pipeline.

        Args:
            model_name: The LLM model to use
            temperature: Temperature setting for the model
            api_key: API key for the model service
            max_workers: Maximum number of concurrent workers
            batch_size: Number of cases to process in each batch
            max_retries: Maximum number of retries for API calls
        """
        self.model = init_chat_model(
            model_name, temperature=temperature, api_key=api_key
        )
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Set up logging
        logger.add("case_analysis_pipeline.log", rotation="100 MB")

        # Load prompts
        self.system_prompt = self._load_system_prompt()
        self.user_prompt_template = self._load_user_prompt_template()

    def _load_system_prompt(self) -> str:
        """Load the system prompt for case analysis"""
        return """You are an expert medical diagnostician and clinical educator with decades of experience. 
Your task is to analyze clinical case vignettes and evaluate them based on specific criteria.
You should focus on diagnostic reasoning, differential diagnoses, and clinical management.
Provide numerical scores (1-10) for each criterion, where 1 is the lowest and 10 is the highest.
Be objective and thorough in your assessment.

Evaluate this case on the following criteria, providing a score from 1-10 for each (where 10 is the highest):

1. DIAGNOSTIC CLARITY (1-10): How clear is the final diagnosis in the vignette? Is there sufficient clinical evidence to support it?

2. DIFFERENTIAL APPROPRIATENESS (1-10): Do the other listed differential diagnoses make sense given the clinical presentation? Are they reasonable alternatives?

3. DIAGNOSTIC SIMILARITY (1-10): How similar is the final diagnosis to the differentials in terms of clinical definition, presentation, and pathophysiology? (Higher score means more distinct diagnoses)

4. MANAGEMENT DIVERGENCE (1-10): How different would the management be between the final diagnosis and the differentials? (Higher score means more divergent management approaches)

5. HARM POTENTIAL (1-10): If the final diagnosis were missed and a differential diagnosis were treated instead, how much potential harm would this cause to the patient? (Higher score means greater potential harm)

For each criterion, provide:
- The numerical score (1-10)
- A brief justification (2-3 sentences)
- Key factors that influenced your scoring decision

Then provide an OVERALL CASE QUALITY SCORE (1-10) that reflects how valuable this case would be for teaching diagnostic reasoning.

Format your response as a JSON object with the following structure:
{
  "diagnostic_clarity": {"score": X, "justification": "...", "key_factors": ["...", "..."]},
  "differential_appropriateness": {"score": X, "justification": "...", "key_factors": ["...", "..."]},
  "diagnostic_similarity": {"score": X, "justification": "...", "key_factors": ["...", "..."]},
  "management_divergence": {"score": X, "justification": "...", "key_factors": ["...", "..."]},
  "harm_potential": {"score": X, "justification": "...", "key_factors": ["...", "..."]},
  "overall_score": X,
  "summary": "A brief summary of why this case is or isn't valuable for teaching diagnostic reasoning."
}"""

    def _load_user_prompt_template(self) -> str:
        """Load the user prompt template for case analysis"""
        return """
Please analyze the following clinical case vignette:
{case_text}
"""

    async def async_model_invoke_with_retry(
        self, messages: List[SystemMessage | HumanMessage]
    ) -> Tuple[AIMessage, int]:
        """
        Asynchronously invoke the language model with retry logic.

        Args:
            messages: List of messages to send to the model

        Returns:
            Tuple of (model response, token count)
        """
        retries = 0
        backoff_time = 1

        while retries <= self.max_retries:
            try:
                response: AIMessage = await self.model.ainvoke(messages)
                return response, response.usage_metadata["total_tokens"]
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Failed after {self.max_retries} retries: {e}")
                    # Return a default error response
                    return (
                        AIMessage(
                            content=json.dumps(
                                {
                                    "error": f"Error after {self.max_retries} retries: {str(e)}"
                                }
                            )
                        ),
                        0,
                    )

                # Exponential backoff
                wait_time = backoff_time * (1.5 ** (retries - 1))
                logger.warning(
                    f"Retry {retries}/{self.max_retries} after error: {e}. Waiting {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)
                backoff_time *= 2

    async def analyze_case(self, case_id: str, case_text: str) -> Dict:
        """
        Analyze a clinical case using the LLM.

        Args:
            case_id: Unique identifier for the case
            case_text: The text of the case to analyze

        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        prompt = self.user_prompt_template.format(case_text=case_text)

        try:
            response, token_count = await self.async_model_invoke_with_retry(
                [SystemMessage(self.system_prompt), HumanMessage(prompt)]
            )

            # Parse JSON response
            result = self._extract_json_from_text(response.content)

            # Add metadata
            result["case_id"] = case_id
            result["processing_time"] = time.time() - start_time
            result["token_count"] = token_count

            logger.info(
                f"Analyzed case {case_id}: Overall score {result.get('overall_score', 'N/A')}"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to analyze case {case_id}: {e}")
            return {
                "case_id": case_id,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def _extract_json_from_text(self, text: str) -> Dict:
        """
        Extract JSON object from text response.

        Args:
            text: Text potentially containing JSON

        Returns:
            Parsed JSON as dictionary
        """
        try:
            # Find JSON object in the response
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
            else:
                # If no JSON found, try to parse the whole response
                return json.loads(text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a structured error
            logger.error(f"Failed to parse JSON from response: {text[:100]}...")
            return {"error": "Failed to parse JSON from response", "raw_response": text}

    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Process a batch of cases in parallel.

        Args:
            batch: List of dictionaries with case_id and text

        Returns:
            List of analysis results
        """
        tasks = []
        for case in batch:
            tasks.append(self.analyze_case(case["case_id"], case["text"]))

        results = await tqdm_asyncio.gather(*tasks, desc="Analyzing cases")
        return results

    async def process_data(
        self, data: pd.DataFrame, text_column: str = "text", id_column: str = "case_id"
    ) -> pd.DataFrame:
        """
        Process all cases in the dataset.

        Args:
            data: DataFrame with cases to analyze
            text_column: Column name containing the case text
            id_column: Column name containing the case ID

        Returns:
            DataFrame with analysis results
        """
        logger.info(f"Starting analysis of {len(data)} cases")

        # Ensure we have a case_id column
        if id_column not in data.columns:
            data["case_id"] = [f"case_{i}" for i in range(len(data))]
            id_column = "case_id"

        # Convert DataFrame to list of dictionaries
        cases = []
        for _, row in data.iterrows():
            cases.append(
                {"case_id": str(row[id_column]), "text": str(row[text_column])}
            )

        # Process in batches
        all_results = []
        for i in range(0, len(cases), self.batch_size):
            batch = cases[i : i + self.batch_size]
            logger.info(
                f"Processing batch {i//self.batch_size + 1}/{(len(cases)-1)//self.batch_size + 1}"
            )

            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)

            # Log progress
            logger.info(f"Completed {len(all_results)}/{len(cases)} cases")

        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)

        # Merge with original data
        merged_df = data.copy()

        # Extract scores and add them to the dataframe
        for i, result in enumerate(all_results):
            case_id = result["case_id"]
            idx = data.index[data[id_column] == case_id].tolist()

            if not idx:
                continue

            idx = idx[0]

            if "error" in result:
                # Handle error cases
                merged_df.loc[idx, "error"] = result.get("error", "Unknown error")
                for criterion in [
                    "diagnostic_clarity",
                    "differential_appropriateness",
                    "diagnostic_similarity",
                    "management_divergence",
                    "harm_potential",
                ]:
                    merged_df.loc[idx, f"{criterion}_score"] = np.nan
                merged_df.loc[idx, "overall_score"] = np.nan
            else:
                # Extract scores
                try:
                    for criterion in [
                        "diagnostic_clarity",
                        "differential_appropriateness",
                        "diagnostic_similarity",
                        "management_divergence",
                        "harm_potential",
                    ]:
                        if criterion in result:
                            merged_df.loc[idx, f"{criterion}_score"] = result[
                                criterion
                            ].get("score", np.nan)
                            merged_df.loc[idx, f"{criterion}_justification"] = result[
                                criterion
                            ].get("justification", "")

                    merged_df.loc[idx, "overall_score"] = result.get(
                        "overall_score", np.nan
                    )
                    merged_df.loc[idx, "summary"] = result.get("summary", "")
                    merged_df.loc[idx, "analysis_json"] = json.dumps(result)
                except Exception as e:
                    merged_df.loc[idx, "error"] = f"Failed to extract scores: {str(e)}"

        # Calculate a weighted composite score
        weights = {
            "diagnostic_clarity_score": 0.2,
            "differential_appropriateness_score": 0.2,
            "diagnostic_similarity_score": 0.2,
            "management_divergence_score": 0.2,
            "harm_potential_score": 0.2,
        }

        score_columns = list(weights.keys())
        merged_df["weighted_score"] = sum(
            merged_df[col] * weight for col, weight in weights.items()
        )

        # Sort by weighted score
        merged_df = merged_df.sort_values("weighted_score", ascending=False)

        logger.success(f"Completed analysis with {len(merged_df)} cases")
        return merged_df

    def save_results(
        self, results_df: pd.DataFrame, output_path: str = "analyzed_cases.csv"
    ):
        """
        Save the analysis results to a CSV file.

        Args:
            results_df: DataFrame with analysis results
            output_path: Path to save the CSV file
        """
        results_df.to_csv(output_path, index=False)
        logger.success(f"Saved {len(results_df)} analyzed cases to {output_path}")

        # Print summary statistics
        logger.info("\nSummary Statistics:")
        for criterion in [
            "diagnostic_clarity_score",
            "differential_appropriateness_score",
            "diagnostic_similarity_score",
            "management_divergence_score",
            "harm_potential_score",
            "overall_score",
            "weighted_score",
        ]:
            if criterion in results_df.columns:
                logger.info(
                    f"{criterion}: Mean = {results_df[criterion].mean():.2f}, Median = {results_df[criterion].median():.2f}"
                )

        # Print top 5 cases by weighted score
        logger.info("\nTop 5 Cases by Weighted Score:")
        top_cases = results_df.head(5)
        for i, row in top_cases.iterrows():
            if "weighted_score" in row and "overall_score" in row:
                logger.info(
                    f"Case {i}: Weighted Score = {row['weighted_score']:.2f}, Overall Score = {row['overall_score']:.2f}"
                )
                if "summary" in row:
                    logger.info(f"Summary: {row['summary'][:200]}...")
                logger.info("-" * 50)


class CompleteCaseProcessingPipeline:
    """
    Complete pipeline that combines filtering and analysis.
    """

    def __init__(
        self,
        model_name: str = "openai:gpt-4.1-mini",
        analysis_model_name: str = "openai:gpt-4.1-mini",
        temperature: float = 0.5,
        analysis_temperature: float = 0.2,
        api_key: str = None,
        total_limit: int = 1000,
        group_limit: int = 150,
        min_score: int = 9,
        batch_size: int = 1000,
        max_concurrent: int = 10,
        analysis_batch_size: int = 10,
        max_workers: int = 4,
    ):
        """
        Initialize the complete case processing pipeline.

        Args:
            model_name: The LLM model to use for filtering
            analysis_model_name: The LLM model to use for analysis
            temperature: Temperature setting for the filtering model
            analysis_temperature: Temperature setting for the analysis model
            api_key: API key for the model service
            total_limit: Maximum total cases to collect
            group_limit: Maximum cases per specialty group
            min_score: Minimum score (1-10) to accept a case
            batch_size: Number of cases to process in each batch for filtering
            max_concurrent: Maximum concurrent API calls for filtering
            analysis_batch_size: Number of cases to process in each batch for analysis
            max_workers: Maximum number of concurrent workers for analysis
        """
        # Initialize the filtering pipeline
        self.filter_pipeline = CaseReportFilterPipeline(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            total_limit=total_limit,
            group_limit=group_limit,
            min_score=min_score,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
        )

        # Initialize the analysis pipeline
        self.analysis_pipeline = ClinicalCaseAnalysisPipeline(
            model_name=analysis_model_name,
            temperature=analysis_temperature,
            api_key=api_key,
            max_workers=max_workers,
            batch_size=analysis_batch_size,
        )

        # Set up logging
        logger.add("complete_pipeline.log", rotation="100 MB")

    async def run_pipeline(
        self,
        data: pd.DataFrame,
        output_filtered_path: str = "filtered_cases.csv",
        output_analyzed_path: str = "analyzed_cases.csv",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete pipeline: filtering followed by analysis.

        Args:
            data: DataFrame with case reports to process
            output_filtered_path: Path to save filtered cases
            output_analyzed_path: Path to save analyzed cases

        Returns:
            Tuple of (filtered_cases, analyzed_cases) DataFrames
        """
        logger.info(f"Starting complete pipeline with {len(data)} cases")

        # Step 1: Filter cases
        logger.info("Step 1: Filtering cases")
        filtered_cases = await self.filter_pipeline.process_data(data)
        self.filter_pipeline.save_results(output_filtered_path)

        # Step 2: Analyze filtered cases
        logger.info(f"Step 2: Analyzing {len(filtered_cases)} filtered cases")
        analyzed_cases = await self.analysis_pipeline.process_data(filtered_cases)
        self.analysis_pipeline.save_results(analyzed_cases, output_analyzed_path)

        logger.success(
            f"Pipeline complete: {len(filtered_cases)} cases filtered, {len(analyzed_cases)} cases analyzed"
        )
        return filtered_cases, analyzed_cases


async def main():
    # Load case reports dataset
    data = pd.read_csv("case_reports.csv")

    # Initialize complete pipeline
    pipeline = CompleteCaseProcessingPipeline(
        model_name="openai:gpt-4.1-mini",
        analysis_model_name="openai:gpt-4-turbo",
        api_key="api-key",
        total_limit=1000,
        group_limit=150,
        min_score=9,
        batch_size=1000,
        max_concurrent=10,
        analysis_batch_size=10,
        max_workers=4,
    )

    # Run the pipeline
    filtered_cases, analyzed_cases = await pipeline.run_pipeline(
        data,
        output_filtered_path="filtered_cases.csv",
        output_analyzed_path="analyzed_cases.csv",
    )

    print(f"Filtered {len(filtered_cases)} cases")
    print(f"Analyzed {len(analyzed_cases)} cases")
    print("\nTop 5 cases by weighted score:")
    top_cases = analyzed_cases.head(5)
    for i, row in top_cases.iterrows():
        print(
            f"Case {i}: Weighted Score = {row['weighted_score']:.2f}, Overall Score = {row['overall_score']:.2f}"
        )
        print(f"Summary: {row['summary'][:200]}...")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
