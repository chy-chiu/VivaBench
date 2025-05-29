import asyncio
import concurrent.futures
import json
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml
from langchain.chat_models import init_chat_model
from loguru import logger
from tqdm import tqdm

from vivabench.examiner import Examination
from vivabench.ontology.schema import ClinicalCase
from vivabench.utils import init_openrouter_chat_model


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_global_logger(level: str):
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def setup_main_loggers(output_dir):
    """Setup the main process logger with a filter to exclude examination logs"""
    main_log_path = os.path.join(output_dir, "main_process.log")

    logger.remove()

    # Add console handler for ERROR level and above (for all logs)
    # This ensures all errors show up in the console regardless of source
    logger.add(
        sys.stderr,
        level="INFO",  # Only show info or above
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        + (
            " | UID: {extra[examination_uid]}" if "examination_uid" in "{extra}" else ""
        ),
    )

    # Add main log handler that excludes examination logs
    main_log_id = logger.add(
        main_log_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        filter=lambda record: "examination_uid" not in record["extra"],
    )

    return logger  # Return the configured logger


def setup_examination_logger(uid, log_dir):
    """Create an examination-specific logger that only logs to its own file"""
    log_path = os.path.join(log_dir, f"{uid}.log")

    # Create a unique handler ID for this examination's log
    handler_id = logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        # This filter ensures ONLY logs for this specific examination go to this file
        filter=lambda record, uid=uid: record["extra"].get("examination_uid") == uid,
        enqueue=True,  # Make logging thread-safe
    )

    # Create a contextualized logger with the examination UID bound to it
    # Every log from this logger will have the examination_uid in its extras
    exam_logger = logger.bind(examination_uid=uid)

    return exam_logger, log_path, handler_id


async def run_single_examination_async(row, agent_model, examiner_model, log_dir):
    """Async version of run_single_examination"""
    uid = row["uid"]
    exam_logger, log_path, logger_id = setup_examination_logger(uid, log_dir)

    start_time = time.time()
    result = {
        "uid": uid,
        "success": False,
        "error_message": "",
        "output_trace": [],
        "output_log_path": str(log_path),
        "exam_output": None,
    }

    try:
        exam_logger.info(f"Starting examination for UID: {uid}")

        c = ClinicalCase.model_validate_json(row["output"])

        # Use async examination - pass the contextualized logger
        exam = Examination(agent_model, c, examiner_model, logger=exam_logger)
        trace, stats = await exam.conduct_examination_async()

        result["success"] = True
        result["output_trace"] = trace
        result["exam_output"] = stats
        exam_logger.info(f"Examination completed successfully for {uid}")

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        exam_logger.error(f"Error in examination: {error_msg}\n{tb}")
        result["error_message"] = error_msg

    runtime = time.time() - start_time
    exam_logger.info(f"Examination completed in {runtime:.2f} seconds")

    # Optional: Clean up the logger for this examination when done
    logger.remove(logger_id)

    return result


async def process_batch_async(
    df_batch, agent_model, examiner_model, log_dir, max_concurrent
):
    """Process a batch of examinations with asyncio for maximum concurrency"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_run_examination(row):
        async with semaphore:
            return await run_single_examination_async(
                row, agent_model, examiner_model, log_dir
            )

    tasks = [bounded_run_examination(row) for _, row in df_batch.iterrows()]

    # Create progress reporting task
    progress = tqdm(total=len(tasks), desc="Examinations")

    results = []
    for future in asyncio.as_completed(tasks):
        result = await future
        results.append(result)
        progress.update(1)
        if result["success"]:
            progress.set_description(f"Latest: {result['uid']} - SUCCESS")
        else:
            progress.set_description(f"Latest: {result['uid']} - FAILED")

    progress.close()
    return results


def run_examinations_async(
    df,
    agent_model,
    examiner_model,
    output_dir="./exam_results",
    max_concurrent=50,
    batch_size=None,
):

    # Create output directories
    output_dir = Path(output_dir)
    log_dir = os.path.join(output_dir, "logs")
    results_dir = os.path.join(output_dir, "results")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    main_log = os.path.join(output_dir, "main_process.log")
    logger.add(
        main_log,
        level="INFO",
        filter=lambda record: "examination_uid" not in record["extra"],
    )

    main_logger = setup_main_loggers(output_dir)

    all_results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Convert model classes to async versions if needed
    if not hasattr(agent_model, "ainvoke"):
        main_logger.warning(
            "Agent model doesn't support async. Performance may be limited."
        )
        # Here you might need to adapt your model to support async operations

    if batch_size:
        total_batches = (len(df) + batch_size - 1) // batch_size
        main_logger.info(
            f"Processing {len(df)} examinations in {total_batches} batches of size {batch_size}"
        )

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            main_logger.info(
                f"Starting batch {i//batch_size + 1}/{total_batches} with {len(batch_df)} examinations"
            )

            # Process each batch with asyncio
            batch_results = asyncio.run(
                process_batch_async(
                    batch_df, agent_model, examiner_model, log_dir, max_concurrent
                )
            )
            all_results.extend(batch_results)

            # Save intermediate results
            batch_results_df = pd.DataFrame(batch_results)
            batch_results_df.to_csv(
                os.path.join(
                    results_dir, f"batch_{i//batch_size + 1}_results_{timestamp}.csv"
                ),
                index=False,
            )
            main_logger.info(f"Completed batch {i//batch_size + 1}")
    else:
        main_logger.info(f"Processing all {len(df)} examinations in a single batch")
        all_results = asyncio.run(
            process_batch_async(
                df, agent_model, examiner_model, log_dir, max_concurrent
            )
        )

    # Create final results dataframe
    results_df = pd.DataFrame(all_results)

    # Save final CSV
    csv_path = os.path.join(results_dir, f"examination_results_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)

    # Save detailed results
    json_results = []
    for r in all_results:
        json_result = r.copy()
        # Convert sets to lists for JSON serialization
        if "exam_output" in json_result and isinstance(
            json_result["exam_output"], dict
        ):
            for k, v in json_result["exam_output"].items():
                if isinstance(v, set):
                    json_result["exam_output"][k] = list(v)
        json_results.append(json_result)

    with open(
        os.path.join(results_dir, f"detailed_results_{timestamp}.json"), "w"
    ) as f:
        json.dump(json_results, f)

    main_logger.info(f"All examinations completed. Results saved to {csv_path}")
    return results_df


def run_single_examination(row, agent_model, examiner_model, log_dir, exam_cfg):
    uid = row["uid"]
    exam_logger, log_path, handler_id = setup_examination_logger(uid, log_dir)

    start = time.time()
    res = dict(
        uid=uid,
        success=False,
        error_message="",
        output_trace=[],
        output_log_path=str(log_path),
        exam_output=None,
    )

    try:
        exam_logger.info(f"Start UID {uid}")
        case = ClinicalCase.model_validate_json(row["output"])

        ex = Examination(
            agent_model=agent_model,
            clincase=case,
            examiner_model=examiner_model,
            examiner_kwargs=dict(
                mapper=exam_cfg["mapper"],
                parser=exam_cfg["parser"],
                hx_limit=exam_cfg["hx_limit"],
                phys_limit=exam_cfg["phys_limit"],
                ix_limit=exam_cfg["ix_limit"],
                img_limit=exam_cfg["img_limit"],
                action_limit=exam_cfg["action_limit"],
                snomed_embeddings_path=exam_cfg["snomed_embeddings_path"],
            ),
            logger=exam_logger,
        )
        trace, stats = ex.conduct_examination()
        res.update(success=True, output_trace=trace, exam_output=stats)
        exam_logger.info("Completed successfully")

    except Exception as e:
        tb = traceback.format_exc()
        exam_logger.error(f"Error: {e}\n{tb}")
        res["error_message"] = str(e)

    runtime = time.time() - start
    exam_logger.info(f"Done in {runtime:.2f}s")
    logger.remove(handler_id)
    return res


def run_examinations_parallel(
    df,
    agent_model,
    examiner_model,
    output_dir="./exam_results",
    max_workers=30,
    batch_size=None,
    examination_config=None,
):

    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    res_dir = output_dir / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    main_logger = setup_main_loggers(output_dir)
    main_logger.info(f"Starting {len(df)} cases with {max_workers} workers")

    all_results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    def _process_batch(batch_df, idx):
        batch_res = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="exam"
        ) as exe:
            futures = {
                exe.submit(
                    run_single_examination,
                    row,
                    agent_model,
                    examiner_model,
                    str(log_dir),
                    examination_config,
                ): row["uid"]
                for _, row in batch_df.iterrows()
            }

            with tqdm(total=len(futures), desc=f"Batch {idx}") as pbar:
                for fut in concurrent.futures.as_completed(futures):
                    uid = futures[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        main_logger.error(f"{uid} executor error: {e}")
                        r = dict(
                            uid=uid,
                            success=False,
                            error_message=str(e),
                            output_trace=[],
                            output_log_path=str(log_dir / f"{uid}.log"),
                            exam_output=None,
                        )
                    batch_res.append(r)
                    status = "OK" if r["success"] else "FAIL"
                    pbar.set_description(f"{uid}→{status}")
                    pbar.update(1)
        return batch_res

    # 1) possibly chunk
    if batch_size:
        n = len(df)
        for i in range(0, n, batch_size):
            sub = df.iloc[i : i + batch_size]
            main_logger.info(f"Batch {i//batch_size+1}: {len(sub)} cases")
            br = _process_batch(sub, i // batch_size + 1)
            all_results.extend(br)
            pd.DataFrame(br).to_csv(
                res_dir / f"batch_{i//batch_size+1}_{timestamp}.csv", index=False
            )
    else:
        all_results = _process_batch(df, 1)

    # 2) save final
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(res_dir / f"eval_results_{timestamp}.csv", index=False)

    with open(res_dir / f"detailed_{timestamp}.json", "w") as f:
        # convert sets to lists
        for rec in all_results:
            if isinstance(rec.get("exam_output"), dict):
                for k, v in rec["exam_output"].items():
                    if isinstance(v, set):
                        rec["exam_output"][k] = list(v)
        json.dump(all_results, f, indent=2)

    main_logger.info("ALL DONE")
    return df_out


def process_batch(df, agent_model, examiner_model, log_dir, max_workers, main_logger):
    """Process a batch of examinations in parallel with optimized thread management"""
    results = []

    # Configure thread pool for optimal performance
    # Setting thread max_workers based on empirical testing
    # Usually slightly less than CPU cores works best for API-bound tasks
    thread_config = {
        "max_workers": max_workers,
        "thread_name_prefix": "exam_worker",
    }

    with concurrent.futures.ThreadPoolExecutor(**thread_config) as executor:
        # Submit all tasks
        future_to_uid = {}
        for _, row in df.iterrows():
            future = executor.submit(
                run_single_examination, row, agent_model, examiner_model, log_dir
            )
            future_to_uid[future] = row["uid"]

        # Track progress with tqdm
        with tqdm(total=len(future_to_uid), desc="Examinations") as progress_bar:
            for future in concurrent.futures.as_completed(future_to_uid):
                uid = future_to_uid[future]

                try:
                    result = future.result()
                    results.append(result)

                    # Update progress
                    status = "SUCCESS" if result["success"] else "FAILED"
                    progress_bar.set_description(f"Latest: {uid} - {status}")
                    progress_bar.update(1)

                except Exception as e:
                    main_logger.error(f"Executor error with {uid}: {str(e)}")
                    results.append(
                        {
                            "uid": uid,
                            "success": False,
                            "error_message": f"Executor error: {str(e)}",
                            "output_trace": [],
                            "output_log_path": str(os.path.join(log_dir, f"{uid}.log")),
                            "exam_output": None,
                        }
                    )
                    progress_bar.update(1)

    return results


if __name__ == "__main__":
    # Load your data
    pubmed_df = pd.read_csv("data_pubmed.csv")

    # Setup your models
    examiner_model = init_chat_model(
        "openai:gpt-4.1", temperature=0, api_key=OPENAI_API
    )
    agent_model = init_openrouter_chat_model(
        "meta-llama/llama-4-maverick", temperature=0, api_key=OPENROUTER_API_KEY
    )

    df = (pubmed_df,)
    agent_model = (agent_model,)
    examiner_model = (examiner_model,)
    output_dir = ("./evaluation_output/llama-4",)

    # Run examinations in parallel
    results_df = run_examinations_parallel(
        config="config.yaml",
        max_workers=30,
        batch_size=100,  # Optional: process in batches
    )

    # Print summary statistics
    success_rate = results_df["success"].mean() * 100
    logger.info(f"Examination success rate: {success_rate:.2f}%")
