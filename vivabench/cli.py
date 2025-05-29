import argparse
import asyncio
import json
import os
import sys
import time

import pandas as pd
import yaml
from langchain.chat_models import init_chat_model
from loguru import logger
from tqdm import tqdm

from vivabench.evaluate import run_examinations_parallel
from vivabench.examiner import Examination
from vivabench.generate import PipelineConfig, run_pipeline
from vivabench.metrics import EvaluationMetrics
from vivabench.ontology.schema import ClinicalCase
from vivabench.utils import init_ollama_chat_model, init_openrouter_chat_model


def setup_global_logger(level: str):
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def init_model_block(mconf: dict):
    prov = mconf["provider"]
    model = mconf["model"]
    temp = float(mconf.get("temperature", 0.0))
    # pick up key either inline or from env
    key = mconf.get("api_key") or os.getenv(mconf.get("api_key_env", ""), None)

    if prov == "openai":
        if not key:
            raise RuntimeError("Missing OpenAI API key")
        return init_chat_model(f"openai:{model}", temperature=temp, api_key=key)

    if prov == "openrouter":
        if not key:
            raise RuntimeError("Missing OpenRouter API key")
        return init_openrouter_chat_model(model, temperature=temp, api_key=key)

    if prov == "ollama":
        host = mconf.get("host", "localhost")
        port = int(mconf.get("port", 11434))
        return init_ollama_chat_model(model, host=host, port=port, temperature=temp)

    raise RuntimeError(f"Unknown provider: {prov}")


def run_metrics(dataset_df, results_df, metrics_args):

    metrics = EvaluationMetrics(**metrics_args)

    eval_results = []

    for _, row in tqdm(results_df[~results_df.exam_output.isna()].iterrows()):
        uid = row["uid"]
        output = dataset_df.loc[uid]["output"]
        output = json.loads(output)

        result = eval(row["exam_output"])
        result["uid"] = uid

        metrics.load_results(
            gt_diagnosis=output["diagnosis"] or [],
            gt_differentials=output["differentials"] or [],
            final_diagnosis=result["final_diagnosis"] or [],
            provisional_diagnosis=result["provisional_diagnosis"] or [],
            full_info_diagnosis=result["full_info_diagnosis"] or [],
            matched_keys=result["matched_keys"],
            unmatched_request_keys=result["unmatched_request_keys"],
            unmatched_case_keys=result["unmatched_case_keys"],
        )

        result.pop("request_log")
        result.update(metrics.compute_all_metrics())
        eval_results.append(result)

    eval_df = pd.DataFrame(eval_results).set_index("uid")
    eval_df = eval_df.join(dataset_df[["vignette", "diagnosis", "differentials"]])

    return eval_df


def do_evaluate(args):
    cfg = load_yaml(args.config)

    # 1) load & override
    if args.evaluation_id:
        cfg["data"]["evaluation_id"] = args.evaluation_id

    model_name = cfg["models"]["agent"]["model"].split("-1")
    input_file = cfg["data"]["input"]

    evaluation_id = (
        cfg["data"]["evaluation_id"] or f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    )

    output_dir = os.path.join(cfg["data"]["output_dir"], evaluation_id)

    if args.input:
        cfg["data"]["input"] = args.input
    if args.output_dir:
        cfg["data"]["output_dir"] = args.output_dir

    # 2) logger
    setup_global_logger(cfg["logging"]["level"])
    logger.info(
        f"Starting evaluation run for model [{model_name}], using dataset [{input_file}. evaluation_id: {evaluation_id}]"
    )

    # 3) data
    df = pd.read_csv(input_file)
    df = df[df.status == "success"]

    # 4) models
    examiner = init_model_block(cfg["models"]["examiner"])
    agent = init_model_block(cfg["models"]["agent"])

    # 5) run evaluation
    df_out = run_examinations_parallel(
        df=df,
        agent_model=agent,
        examiner_model=examiner,
        output_dir=output_dir,
        max_workers=cfg["data"]["max_workers"],
        batch_size=cfg["data"]["batch_size"],
        examination_config=cfg["examination"],
    )

    rate = df_out["success"].mean() * 100
    logger.info(
        f"Evaluation finished for {evaluation_id}: success rate {rate:.2f}%. Evaluation logs at {output_dir}. Now calculating metrics.."
    )

    df = df.set_index("uid")
    # 6) run metrics
    eval_df = run_metrics(df, df_out, cfg["metrics"])
    eval_df.to_csv(os.path.join(output_dir, "metrics.csv"))


def do_metrics(args):

    cfg = load_yaml(args.config)
    output_filepath = args.output_csv
    df_out = pd.read_csv(output_filepath)

    output_dir = (
        os.path.join(output_filepath.split("/")[:-1])
        if "/" in df_out
        else cfg["data"].get("output_dir", "./")
    )

    df = pd.read_csv(cfg["data"]["input"])
    df = df[df.status == "success"]

    df = df.set_index("uid")
    eval_df = run_metrics(df, df_out, cfg["metrics"])
    eval_df.to_csv(os.path.join(output_dir, "metrics.csv"))


def do_generate(args):
    # 1) load & override
    cfg = load_yaml(args.config)

    if args.input:
        cfg["pipeline"]["input"] = args.input
    if args.output:
        cfg["pipeline"]["output"] = args.output

    # 2) logger
    setup_global_logger(cfg["logging"]["level"])
    logger.info("Starting GENERATION run…")

    # 3) build PipelineConfig
    pc = PipelineConfig(
        input_path=cfg["pipeline"]["input"],
        output_path=cfg["pipeline"]["output"],
        snomed_embedding_path=cfg["embeddings"]["snomed"],
        icd_embedding_path=cfg["embeddings"]["icd_embedding"],
        icd_mapping_path=cfg["mappings"]["icd_map"],
        model_name=cfg["models"]["generator"]["model"],
        reasoning_model_name=cfg["models"]["reasoning"]["model"],
        batch_size=cfg["pipeline"]["batch_size"],
        limit=cfg["pipeline"]["limit"],
        api_key=(
            cfg["models"]["generator"].get("api_key")
            or os.getenv(cfg["models"]["generator"]["api_key_env"], "")
        ),
    )

    # 4) run the async pipeline
    asyncio.run(run_pipeline(pc))
    logger.info("Generation finished. Running metrics now...")


def main():
    parser = argparse.ArgumentParser(
        prog="vivabench", description="VivaBench: evaluate or generate clinical cases"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── EVALUATE ────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Run evaluation on VivaBench dataset")
    p_eval.add_argument(
        "-c",
        "--config",
        default="configs/evaluate.yaml",
        help="Path to evaluation config YAML",
    )
    p_eval.add_argument("--input", help="Override input CSV path")
    p_eval.add_argument("--output_dir", help="Override output directory")
    p_eval.add_argument("--evaluation_id", help="ID to identify this evaluation run")

    # ── GENERATE ────────────────────────────────────────────────────────────────
    p_gen = sub.add_parser(
        "generate", help="Generate new cases from clinical vignettes"
    )
    p_gen.add_argument(
        "-c",
        "--config",
        default="configs/generate.yaml",
        help="Path to generation config YAML",
    )
    p_gen.add_argument(
        "--input", help="Override input CSV path for input clinical vignettes"
    )
    p_gen.add_argument(
        "--output", help="Override output CSV path for generation artifact"
    )

    # ── METIRCS ──────────────────────────────────────────────────────────────────
    p_met = sub.add_parser("metrics", help="Re-run metrics on output df")
    p_met.add_argument(
        "-c",
        "--config",
        default="configs/evaluate.yaml",
        help="Path to evaluation config YAML",
    )
    p_met.add_argument(
        "--output_csv", required=True, help="Path to evaluation output CSV"
    )
    p_met.add_argument("--output_dir", help="Path to evaluation output directory")

    args = parser.parse_args()

    if args.command == "evaluate":
        do_evaluate(args)
    elif args.command == "generate":
        do_generate(args)
    elif args.command == "metrics":
        do_metrics(args)


if __name__ == "__main__":
    main()
