"""
Clinical Case Generator Pipeline

This script processes a DataFrame of clinical cases from vignettes and generates
structured clinical cases using an AI model.
"""

import argparse
import asyncio
import json
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import pandas as pd
import txtai
from langchain.chat_models import init_chat_model
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from vivabench.generator import CaseGenerator
from vivabench.ontology.schema import ClinicalCase

# Configure logger
logger.remove()
logger.add(
    "case_generation_{time}.log",
    rotation="100 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")


class PipelineConfig(BaseModel):
    """Configuration for the pipeline."""

    input_path: str
    output_path: str
    snomed_embedding_path: str = "./medical/snomed_embeddings"
    icd_embedding_path: str = "./medical/icd_embeddings"
    icd_mapping_path: str = "./medical/d_icd_diagnoses.csv"
    model_name: str = "openai:gpt-4.1"
    reasoning_model_name: str = "openai:o4-mini"
    batch_size: int = 10
    limit: Optional[int] = None
    api_key: str = ""


async def process_batch(
    batch: pd.DataFrame,
    generator,
) -> List[Dict[Any, Any]]:
    """Process a batch of cases asynchronously."""
    tasks = []

    for _, row in batch.iterrows():
        # Prepare the vignette by concatenating title and text
        # vignette = f"TITLE: {row['title']}\nCASE: {row['text']}"
        vignette = row["vignette"]
        # uid = row['PMID']
        uid = row["uid"]
        diagnosis = row["diagnosis"]
        differentials = row["differentials"]

        # Create a task for each case
        task = asyncio.create_task(
            process_single_case(
                generator=generator,
                vignette=vignette,
                diagnosis=diagnosis,
                differentials=differentials,
                uid=uid,
            )
        )
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Error processing case: {str(result)}")
            continue
        processed_results.append(result)

    return processed_results


async def process_single_case(
    generator: CaseGenerator,
    vignette: str,
    diagnosis: str,
    differentials: str,
    uid: str,
) -> Dict[Any, Any]:
    """Process a single case and handle any errors."""
    # Create base record with input data
    record = {
        "uid": uid,
        "vignette": vignette,
        # "diagnosis": diagnosis,
        # "differentials": differentials,
        "diagnosis": [],
        "differentials": [],
    }

    # Generate the case
    result = await generator.generate_case(
        vignette=vignette, diagnosis=diagnosis, differentials=differentials
    )

    # Update the record with the result
    record.update(result)

    # Log success or error
    if result["status"] == "success":
        # Validate and get the full prompt
        try:
            clinical_case = ClinicalCase.model_validate_json(result["output"])
            logger.info(f"Successfully processed case {uid}")
            logger.debug(f"Full prompt for {uid}:\n{clinical_case.full_information}")
        except Exception as e:
            logger.exception(
                f"Successfully created case, but somehow unable to parse case: {e}"
            )
    else:
        logger.warning(
            f"Failed to process case {uid}, {result['status']}: {result['error_message']}"
        )

    record["diagnosis"] = json.loads(result["artifact"]).get("validated_diagnoses", "")
    record["differentials"] = json.loads(result["artifact"]).get(
        "validated_differentials", ""
    )

    return record


async def run_pipeline(config: PipelineConfig):
    """Run the full pipeline."""
    start_time = time.time()
    logger.info(f"Starting pipeline with config: {config}")

    # Load the data
    logger.info(f"Loading data from {config.input_path}")
    df = pd.read_csv(config.input_path)

    if config.limit:
        df = df.head(config.limit)
        logger.info(f"Limited to {config.limit} rows")

    logger.info(f"Loaded {len(df)} cases")

    # Initialize the model
    logger.info(f"Initializing model: {config.model_name}")
    logger.info(f"Initializing reasoning model: {config.reasoning_model_name}")

    os.environ["OPENAI_API_KEY"] = config.api_key
    model = init_chat_model(config.model_name)
    reasoning_model = init_chat_model(config.reasoning_model_name)

    # Initialize the generator
    generator = CaseGenerator(
        model=model, reasoning_model=reasoning_model
    )  # , snomed_embeddings=snomed_embeddings)

    # Process in batches
    all_results = []
    batches = [
        df[i : i + config.batch_size] for i in range(0, len(df), config.batch_size)
    ]
    logger.info(f"Processing {len(batches)} batches of size {config.batch_size}")

    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)}")
        batch_results = await process_batch(batch, generator)
        all_results.extend(batch_results)

        # Force garbage collection after each batch
        import gc

        gc.collect()

        # Convert results to DataFrame and save
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(config.output_path, index=False)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(config.output_path, index=False)
    logger.info(f"Saved {len(results_df)} results to {config.output_path}")

    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
    logger.info(
        f"Successful cases: {sum(1 for r in all_results if r['status'] == 'success')}"
    )
    logger.info(
        f"Failed cases: {sum(1 for r in all_results if r['status'] != 'success')}"
    )


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Clinical Case Generator Pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--snomed",
        default="./medical/snomed_embeddings",
        help="Path to SNOMED-CT embeddings",
    )
    parser.add_argument(
        "--icd-embedding",
        default="./medical/icd_embeddings",
        help="Path to ICD-10 embeddings",
    )
    parser.add_argument(
        "--icd-map",
        default="./medical/d_icd_diagnoses.csv",
        help="Path to ICD-10 codes",
    )
    parser.add_argument("--model", default="openai:gpt-4.1", help="Model name")
    parser.add_argument(
        "--reasoning-model", default="openai:o4-mini", help="Reasoning Model name"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for processing"
    )
    parser.add_argument("--limit", type=int, help="Limit number of rows to process")
    parser.add_argument(
        "--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )

    args = parser.parse_args()

    # Create config
    config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        snomed_embedding_path=args.snomed,
        icd_embedding_path=args.icd_embedding,
        icd_mapping_path=args.icd_map,
        model_name=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY", ""),
    )

    # Run the pipeline
    asyncio.run(run_pipeline(config))


if __name__ == "__main__":
    main()
