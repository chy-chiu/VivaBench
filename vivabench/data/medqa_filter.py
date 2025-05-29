# Outdated script to evaluate appropriate cases from MedQA. Included for reference

import asyncio
import json
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from loguru import logger
from tqdm.asyncio import tqdm_asyncio


class MedQAuestionEvaluationPipeline:
    def __init__(
        self,
        model_name: str = "openai:gpt-4.1-mini",
        temperature: float = 0.5,
        api_key: str = None,
        batch_size: int = 20,
        max_concurrent: int = 20,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        augmented=True,
    ):
        """
        Initialize the medical case evaluation pipeline.

        Args:
            model_name: The LLM model to use
            temperature: Temperature setting for the model
            api_key: API key for the model service
            batch_size: Number of cases to process in each batch
            max_concurrent: Maximum number of concurrent API calls
            max_retries: Maximum number of retries for failed API calls
            retry_delay: Delay between retries in seconds
        """
        self.model = init_chat_model(
            model_name, temperature=temperature, api_key=api_key
        )
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Semaphore to control concurrency
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Set up logging
        logger.add("medical_case_evaluation.log", rotation="100 MB")

        # Track statistics
        self.stats = {
            "total_processed": 0,
            "accepted": 0,
            "rejected": 0,
            "errors": 0,
            "total_time": 0,
            "avg_time_per_case": 0,
        }
        if augmented:
            self.system_prompt = """You are an expert medical diagnostician and educator specializing in clinical reasoning. Your task is to evaluate synthetic medical cases derived from USMLE questions, focusing specifically on the quality and educational value of the case.

Analyze the provided synthetic case thoroughly, considering both ORIGINAL and AUGMENTED information. Your evaluation must focus on clinical relevance, diagnostic reasoning, and educational utility.

## Evaluation Categories
Score each category from 1-10 (where 10 is highest) and provide detailed justification:

### 1. DIAGNOSIS RELEVANCE (1-10)
- Is the diagnosis clinically mainstream and encountered by general practitioners?
- Is it overly niche or requiring subspecialist expertise?
- Is it an extremely rare condition or random anatomical variant?

### 2. DIAGNOSTIC JOURNEY QUALITY (1-10)
- Does the case focus appropriately on the diagnostic process rather than treatment?
- Would the case challenge and educate clinicians about important diagnostic considerations?
- Does it represent a valuable learning opportunity (e.g., commonly missed diagnosis)?

### 3. ORIGINAL DATA SUFFICIENCY (1-10)
- Is the diagnosis plausible based ONLY on the ORIGINAL history, exam, and investigations?
- Would a competent clinician reasonably consider this diagnosis with only the original data?
- Are critical diagnostic clues present in the original information?

### 4. AUGMENTED DATA QUALITY (1-10)
- Do the AUGMENTED history/examination/investigations align with the diagnosis?
- Are the additions clinically coherent and realistic?
- Do the augmentations enhance the educational value without making diagnosis too obvious?

### 5. CLINICAL COHERENCE (1-10)
- Is there internal consistency between all elements of the case?
- Do the clinical features logically fit together?
- Are there any contradictions or implausibilities?

## Differential Diagnosis Analysis
Provide 3-5 reasonable differential diagnoses given the clinical presentation, ranked by likelihood, with brief justification for each.

## Output Format
{
  "reasoning": "your reasoning for score. put all your thinking here",
  "diagnosis_relevance": int,
  "diagnostic_journey_quality": int,
  "original_data_sufficiency": int,
  "augmented_data_quality": int,
  "clinical_coherence": int,
  "overall_score": int,
  "recommendation": "ACCEPT" or "REJECT",
  "differential_diagnoses": ["other diagnosis to be considered"]
}

The case should be ACCEPTED if overall score is ≥8. You need to be as harsh as you can."""
        else:
            self.system_prompt = """You are an expert medical diagnostician and educator specializing in clinical reasoning. Your task is to evaluate synthetic medical cases derived from USMLE questions, focusing specifically on the quality and educational value of the case.

Analyze the provided USMLE question vignette thoroughly. Your evaluation must focus on clinical relevance, diagnostic reasoning, and educational utility.

## Evaluation Categories
Score each category from 1-10 (where 10 is highest) and provide detailed justification:

### 1. DIAGNOSIS RELEVANCE (1-10)
- Is the diagnosis clinically mainstream and encountered by general practitioners?
- Is it overly niche or requiring subspecialist expertise?
- Is it an extremely rare condition or random anatomical variant?
- Additionally, is the provided diagnosis correct and consistent?

### 2. DIAGNOSTIC JOURNEY QUALITY (1-10)
- Does the case focus appropriately on the diagnostic process rather than treatment?
- Would the case challenge and educate clinicians about important diagnostic considerations?
- Does it represent a valuable learning opportunity (e.g., commonly missed diagnosis)?

### 3. DATA SUFFICIENCY (1-10)
- Is the diagnosis plausible based ONLY on the history, exam, and investigations / imaging?
- Would a competent clinician reasonably consider this diagnosis with only the original data?
- Are critical diagnostic clues present in the original information?

### 4. CLINICAL COHERENCE (1-10)
- Is there internal consistency between all elements of the case?
- Do the clinical features logically fit together?
- Are there any contradictions or implausibilities?

## Differential Diagnosis Analysis
Provide 3 reasonable differential diagnoses given the clinical presentation.

## Specialty group analysis
Describe which specialty group this question falls under. Classify into one of these groups: "Cardiovascular & Metabolic", "Respiratory", "Gastrointestinal", "Musculoskeletal & Pain", "Neurological / Psychiatric", "Infectious Disease & Immunology", "Endocrine & Reproductive", "Pediatric", "Other"

## Output Format
{
  "reasoning": "your reasoning for score. put all your thinking here",
  "diagnosis_relevance": int,
  "diagnostic_journey_quality": int,
  "original_data_sufficiency": int,
  "clinical_coherence": int,
  "overall_score": int,
  "recommendation": "ACCEPT" or "REJECT",
  "diagnosis": "your diagnosis"
  "specialty_group": "the specialty group this question should go under"
  "differential_diagnoses": ["other diagnosis to be considered"]
}

The case should be ACCEPTED if overall score is ≥8. You need to be as harsh as you can."""

    def _load_system_prompt(self) -> str:
        """Load the system prompt for case evaluation"""
        return self.system_prompt

    async def async_model_invoke(
        self, messages: List[SystemMessage | HumanMessage], retry_count: int = 0
    ) -> Tuple[AIMessage, int]:
        """
        Asynchronously invoke the language model with retry logic.

        Args:
            messages: List of messages to send to the model
            retry_count: Current retry attempt

        Returns:
            Tuple of (model response, token count)
        """
        async with self.semaphore:
            try:
                response: AIMessage = await self.model.ainvoke(messages)
                return response, response.usage_metadata["total_tokens"]
            except Exception as e:
                if retry_count < self.max_retries:
                    logger.warning(
                        f"Retrying after error: {e} (attempt {retry_count + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(
                        self.retry_delay * (retry_count + 1)
                    )  # Exponential backoff
                    return await self.async_model_invoke(messages, retry_count + 1)
                else:
                    logger.error(
                        f"Error invoking model after {self.max_retries} retries: {e}"
                    )
                    # Return a default error response
                    error_response = {
                        "reasoning": f"Error: {str(e)}",
                        "diagnosis_relevance": 0,
                        "diagnostic_journey_quality": 0,
                        "original_data_sufficiency": 0,
                        "augmented_data_quality": 0,
                        "clinical_coherence": 0,
                        "overall_score": 0,
                        "recommendation": "REJECT",
                        "differential_diagnoses": [],
                        "error": str(e),
                    }
                    return AIMessage(content=json.dumps(error_response)), 0

    def _extract_json_from_text(self, text: str) -> Dict:
        """
        Extract JSON object from text response.

        Args:
            text: Text potentially containing JSON

        Returns:
            Parsed JSON as dictionary
        """
        try:
            # Try to extract JSON if surrounded by markdown code blocks
            json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            matches = re.findall(json_pattern, text)
            if matches:
                return json.loads(matches[0])

            # Try to find JSON object in the response
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)

            # If no JSON found, try to parse the whole response
            return json.loads(text)
        except json.JSONDecodeError:
            # If JSON parsing fails, attempt to fix common issues
            try:
                # Replace single quotes with double quotes
                fixed_text = text.replace("'", '"')
                return json.loads(fixed_text)
            except:
                # If all parsing attempts fail, return a structured error
                logger.error(f"Failed to parse JSON from response: {text[:200]}...")
                return {
                    "reasoning": f"Failed to parse JSON from response: {text[:200]}...",
                    "diagnosis_relevance": 0,
                    "diagnostic_journey_quality": 0,
                    "original_data_sufficiency": 0,
                    "augmented_data_quality": 0,
                    "clinical_coherence": 0,
                    "overall_score": 0,
                    "recommendation": "REJECT",
                    "differential_diagnoses": [],
                    "error": "JSON parsing failed",
                    "raw_response": text,
                }

    async def evaluate_case(self, case_id: str, case_text: str) -> Dict:
        """
        Evaluate a medical case using the LLM.

        Args:
            case_id: Unique identifier for the case
            case_text: The text of the case to evaluate

        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()

        try:
            response, token_count = await self.async_model_invoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=case_text),
                ]
            )

            # Parse JSON response
            result = self._extract_json_from_text(response.content)

            # Add metadata
            result["case_id"] = case_id
            result["processing_time"] = time.time() - start_time
            result["token_count"] = token_count

            # Log result
            status = result.get("recommendation", "UNKNOWN")
            score = result.get("overall_score", "N/A")
            logger.info(f"Evaluated case {case_id}: Score {score}, {status}")

            return result

        except Exception as e:
            logger.error(f"Failed to evaluate case {case_id}: {e}")
            error_response = {
                "case_id": case_id,
                "reasoning": f"Error during evaluation: {str(e)}",
                "diagnosis_relevance": 0,
                "diagnostic_journey_quality": 0,
                "original_data_sufficiency": 0,
                "augmented_data_quality": 0,
                "clinical_coherence": 0,
                "overall_score": 0,
                "recommendation": "REJECT",
                "differential_diagnoses": [],
                "error": str(e),
                "processing_time": time.time() - start_time,
            }
            return error_response

    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Process a batch of cases in parallel.

        Args:
            batch: List of dictionaries with case_id and text

        Returns:
            List of evaluation results
        """
        tasks = []
        for case in batch:
            tasks.append(self.evaluate_case(case["case_id"], case["text"]))

        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating cases")

        # Update statistics
        for result in results:
            self.stats["total_processed"] += 1
            if "error" in result:
                self.stats["errors"] += 1
            elif result.get("recommendation") == "ACCEPT":
                self.stats["accepted"] += 1
            else:
                self.stats["rejected"] += 1

        return results

    async def process_data(
        self,
        data: pd.DataFrame,
        text_column: str,
        id_column: str = None,
        output_path: str = None,
    ) -> pd.DataFrame:
        """
        Process all cases in the dataset.

        Args:
            data: DataFrame with cases to evaluate
            text_column: Column name containing the case text
            id_column: Column name containing the case ID (optional)
            output_path: Path to save intermediate results (optional)

        Returns:
            DataFrame with evaluation results
        """
        start_time = time.time()
        logger.info(f"Starting evaluation of {len(data)} medical cases")

        # Ensure we have a case_id column
        if id_column is None or id_column not in data.columns:
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
        total_batches = (len(cases) - 1) // self.batch_size + 1

        for i in range(0, len(cases), self.batch_size):
            batch = cases[i : i + self.batch_size]
            current_batch = i // self.batch_size + 1
            logger.info(f"Processing batch {current_batch}/{total_batches}")

            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)

            # Calculate and log progress statistics
            accepted = sum(
                1 for r in batch_results if r.get("recommendation") == "ACCEPT"
            )
            rejected = sum(
                1 for r in batch_results if r.get("recommendation") == "REJECT"
            )
            errors = sum(1 for r in batch_results if "error" in r)

            logger.info(
                f"Batch {current_batch} results: {accepted} accepted, {rejected} rejected, {errors} errors"
            )
            logger.info(
                f"Overall progress: {len(all_results)}/{len(cases)} cases processed"
            )

            # Save intermediate results if output path is provided
            if output_path:
                interim_results_df = pd.DataFrame(all_results)
                interim_results_df.to_csv(
                    f"{output_path}_interim_{current_batch}.csv", index=False
                )
                logger.info(
                    f"Saved interim results to {output_path}_interim_{current_batch}.csv"
                )

        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)

        # Calculate overall statistics
        self.stats["total_time"] = time.time() - start_time
        self.stats["avg_time_per_case"] = (
            self.stats["total_time"] / len(data) if len(data) > 0 else 0
        )

        # Log overall statistics
        logger.success(
            f"Completed evaluation of {len(data)} cases in {self.stats['total_time']:.2f} seconds"
        )
        logger.info(
            f"Average time per case: {self.stats['avg_time_per_case']:.2f} seconds"
        )
        logger.info(
            f"Accepted: {self.stats['accepted']} ({self.stats['accepted']/len(data)*100:.1f}%)"
        )
        logger.info(
            f"Rejected: {self.stats['rejected']} ({self.stats['rejected']/len(data)*100:.1f}%)"
        )
        logger.info(
            f"Errors: {self.stats['errors']} ({self.stats['errors']/len(data)*100:.1f}%)"
        )

        return results_df

    def run(
        self,
        data: pd.DataFrame,
        text_column: str,
        id_column: str = None,
        output_path: str = "evaluated_cases.csv",
    ) -> pd.DataFrame:
        """
        Run the full evaluation pipeline.

        Args:
            data: DataFrame with cases to evaluate
            text_column: Column name containing the case text
            id_column: Column name containing the case ID (optional)
            output_path: Path to save the results

        Returns:
            DataFrame with evaluation results
        """
        try:
            # Create event loop if not exists
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results_df = loop.run_until_complete(
            self.process_data(data, text_column, id_column, output_path)
        )

        # Save final results
        results_df.to_csv(output_path, index=False)
        logger.success(f"Saved {len(results_df)} evaluated cases to {output_path}")

        return results_df

    def save_statistics(self, output_path: str = "evaluation_stats.json"):
        """
        Save the evaluation statistics to a JSON file.

        Args:
            output_path: Path to save the statistics
        """
        with open(output_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Saved evaluation statistics to {output_path}")

    @staticmethod
    def combine_case_columns(
        df: pd.DataFrame,
        history_col: str = "history_input",
        physical_col: str = "physical_input",
        investigations_col: str = "investigations_input",
        imaging_col: str = "imaging",
        additional_imaging_col: str = None,
    ) -> pd.Series:
        """
        Combine multiple case-related columns into a single text.

        Args:
            df: DataFrame containing the case data
            history_col: Column name for history
            physical_col: Column name for physical examination
            investigations_col: Column name for investigations
            imaging_col: Column name for imaging
            additional_imaging_col: Column name for additional imaging (optional)

        Returns:
            Series of combined case texts
        """
        combined = []
        for _, row in df.iterrows():
            case_text = ""
            if history_col in df.columns and not pd.isna(row[history_col]):
                case_text += str(row[history_col]) + "\n\n"
            if physical_col in df.columns and not pd.isna(row[physical_col]):
                case_text += str(row[physical_col]) + "\n\n"
            if investigations_col in df.columns and not pd.isna(
                row[investigations_col]
            ):
                case_text += str(row[investigations_col]) + "\n\n"
            if imaging_col in df.columns and not pd.isna(row[imaging_col]):
                case_text += str(row[imaging_col]) + "\n\n"
            if (
                additional_imaging_col
                and additional_imaging_col in df.columns
                and not pd.isna(row[additional_imaging_col])
            ):
                case_text += str(row[additional_imaging_col])

            combined.append(case_text.strip())

        return pd.Series(combined)
