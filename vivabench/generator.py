import asyncio
import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import rapidjson
import txtai
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel

from vivabench.ontology.schema import (
    ClinicalCase,
    Demographics,
    Differential,
    History,
    ImagingResult,
    Investigations,
    PhysicalExamination,
    Symptom,
)
from vivabench.prompts.generate import (
    ADDIT_HISTORY_PARSE,
    DDX_PROMPT,
    DDX_VALIDATION_PROMPT,
    DDX_VALIDATION_SYSTEM,
    HOPC_SPLIT_PROMPT,
    HOPC_SPLIT_SYSTEM,
    HX_CLEAN_PROMPT,
    IMAGING_PARSE_PROMPT,
    INVESTIGATION_PARSE_PROMPT,
    PHYSICAL_PARSE_PROMPT,
    VIGNETTE_TRANSFORM_PROMPT,
    VIGNETTE_TRANSFORM_SYSTEM,
)
from vivabench.prompts.generate_medqa import (
    HISTORY_PARSE_AUGMENTED,
    MEDQA_HISTORY_EXPAND_PROMPT,
    MEDQA_IMAGING_PARSE_EXPAND_PROMPT,
    MEDQA_INVESTIGATION_EXPAND_PROMPT,
    MEDQA_INVESTIGATION_PARSE_PROMPT,
    MEDQA_PHYSICAL_EXPAND_PROMPT,
    PHYSICAL_PARSE_AUG_PROMPT,
    SYMPTOMS_PARSE_AUGMENTED,
)
from vivabench.utils import remove_json_markdown


class GenerationResult(BaseModel):

    status: str = ""
    error_message: str = ""
    tokens: int = 0
    output: Union[Dict[str, Any], str] = {}
    artifact: Union[Dict[str, Any], str] = {}

    def model_dump(self):

        if self.output:
            self.output = json.dumps(self.output)
        if self.artifact:
            self.artifact = json.dumps(self.artifact)

        return super().model_dump()


class CaseGenerator:

    def __init__(
        self,
        model: BaseChatModel,
        reasoning_model: BaseChatModel = None,
        output_file: str = "",
        snomed_embedding_path="./medical/snomed_embeddings",
        icd_embedding_path="./medical/icd_embeddings",
        icd_mapping_path="./medical/d_icd_diagnoses.csv",
    ):
        """
        Initialize the CaseGenerator with a language model and SNOMED-CT embeddings.

        Args:
            model: The language model to use for text generation
            output_file: Optional file path to save results
            snomed_embedding_path: Path to preloaded SNOMED embeddings
            icd_embedding_path: Path to preloaded ICD-10 embeddings
            icd_embedding_path: Path to preloaded ICD-10 mapping
        """
        self.model = model
        if reasoning_model:
            self.reasoning_model = reasoning_model
        else:
            self.reasoning_model = model
        self.output_file = output_file

        if snomed_embedding_path:
            logger.info("Using preloaded embeddings for SNOMED")
            self.snomed_embeddings = txtai.Embeddings(
                path="neuml/pubmedbert-base-embeddings", content=True
            )
            self.snomed_embeddings.load(snomed_embedding_path)
        else:
            raise ValueError("need SNOMED embeddings path")

        if icd_embedding_path:
            logger.info("Using preloaded embeddings for icd-10 mapping")
            self.icd10_embeddings = txtai.Embeddings(
                path="neuml/pubmedbert-base-embeddings", content=True
            )
            self.icd10_embeddings.load(icd_embedding_path)
        else:
            raise ValueError("need ICD-10 embeddings path")

        if icd_mapping_path:
            self.icd10 = pd.read_csv(icd_mapping_path).query("icd_version==10")
            self.icd10_mapping = self.icd10.set_index("icd_code").long_title.to_dict()
        else:
            raise ValueError("need ICD-10 mapping path")

    async def async_model_invoke(
        self, messages: List[SystemMessage | HumanMessage], use_reasoning=False
    ) -> Tuple[AIMessage, int]:
        """
        Asynchronously invoke the language model.

        Args:
            messages: List of messages to send to the model

        Returns:
            Tuple of (model response, token count)
        """
        if use_reasoning:
            response: AIMessage = await self.reasoning_model.ainvoke(messages)
        else:
            response: AIMessage = await self.model.ainvoke(messages)

        return response, response.usage_metadata["total_tokens"]

    async def parse_diagnosis(self, result: GenerationResult, vignette: str):

        ddx_tokens = 0

        try:

            ddx_response, ddx_tokens = await self.async_model_invoke(
                [SystemMessage(DDX_PROMPT), HumanMessage(vignette)], use_reasoning=True
            )
            result.tokens += ddx_tokens

            parse_diagnosise_response_raw = remove_json_markdown(ddx_response.content)
            result.artifact["parse_diagnosise_response_raw"] = (
                parse_diagnosise_response_raw
            )

            ddx = rapidjson.loads(
                parse_diagnosise_response_raw,
                parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
            )

            _primary = []
            _differentials = []

            for d in ddx["primary_diagnoses"]:

                icd10_code = d["icd10_code"]
                icd10_desc = d["icd10_description"]

                matched_icd10_code, matched_icd10_desc = self.parse_icd10_code(
                    icd10_code, icd10_desc
                )

                d["icd10_code"] = matched_icd10_code
                d["icd10_description"] = matched_icd10_desc

                _primary.append(str(d))

            for d in ddx["differential_diagnoses"]:

                if d.get("confidence", "Low") == "Low":
                    continue

                icd10_code = d["icd10_code"]
                icd10_desc = d["icd10_description"]

                matched_icd10_code, matched_icd10_desc = self.parse_icd10_code(
                    icd10_code, icd10_desc
                )

                d["icd10_code"] = matched_icd10_code
                d["icd10_description"] = matched_icd10_desc

                _primary.append(str(d))

            ddx["primary_diagnoses"] = _primary
            ddx["differential_diagnoses"] = _differentials

            return ddx, result

        except Exception as e:
            logger.warning("Unable to parse diagnosis from vignette")
            logger.exception(e)
            return None, result

    def parse_icd10_code(self, icd10_code, icd10_desc):
        APPROX_THRESHOLD = 0.7
        EXACT_THRESHOLD = 0.99

        _icd10_code = icd10_code.replace(".", "")

        icd_description_matches = self.icd10_embeddings.search(icd10_desc)

        matched_icd10_desc = ""
        matched_icd10_code = ""
        for m in icd_description_matches:
            if m["score"] > EXACT_THRESHOLD:
                matched_icd10_desc = m["text"]
                matched_icd10_code = self.icd10.icd_code.to_list()[int(m["id"])]

        _icd10_desc_from_code = self.icd10_mapping.get(_icd10_code)
        for m in icd_description_matches:
            if _icd10_desc_from_code == m["text"]:
                matched_icd10_desc = m["text"]
                matched_icd10_code = _icd10_code

        else:
            c = [m for m in icd_description_matches if m["score"] > APPROX_THRESHOLD]
            if c:
                m = c[0]
                matched_icd10_desc = m["text"]
                matched_icd10_code = self.icd10.icd_code.to_list()[int(m["id"])]

        if not matched_icd10_code or not matched_icd10_desc:
            logger.warning(f"Unable to parse ICD-10: {icd10_code} {icd10_desc}")

        return matched_icd10_code, matched_icd10_desc

    def snomed_to_key(self, snomed_term: str) -> str:
        """
        Convert a SNOMED term to a standardized key format.

        Args:
            snomed_term: SNOMED term to convert

        Returns:
            Standardized key
        """
        # Simple implementation - could be enhanced
        return snomed_term.lower().replace("'", "").replace(" ", "_").replace("-", "_")

    async def transform_vignette(self, vignette: str) -> Tuple[Dict[str, str], int]:
        """
        Transform an unstructured vignette into grouped free-text sections.

        Args:
            vignette: Unstructured clinical vignette

        Returns:
            Tuple of (grouped vignette sections, token count)
        """
        response, tokens = await self.async_model_invoke(
            [
                SystemMessage(VIGNETTE_TRANSFORM_SYSTEM),
                HumanMessage(VIGNETTE_TRANSFORM_PROMPT.format(vignette=vignette)),
            ]
        )

        vignette_grouped = rapidjson.loads(
            remove_json_markdown(response.content),
            parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
        )
        return vignette_grouped, tokens

    async def process_vignette(
        self, vignette: str, diagnosis: List[str] = [], differentials: List[str] = []
    ) -> GenerationResult:
        """
        Process an unstructured vignette into a structured ClinicalCase.

        Args:
            vignette: Unstructured clinical vignette
            diagnosis: Optional diagnosis

        Returns:
            Dictionary with processing results and structured case
        """
        total_tokens = 0
        artifact = {
            "vignette": vignette,
            "diagnosis": diagnosis,
            "differentials": differentials,
        }

        if diagnosis:
            vignette += f"\nDiagnosis: {diagnosis}"

        # Step 1: Transform vignette into sections
        try:
            vignette_grouped, transform_tokens = await self.transform_vignette(vignette)
            total_tokens += transform_tokens
            artifact["vignette_grouped"] = vignette_grouped

            # For structured history
            hopc = f"Demographics: {vignette_grouped['demographics']}\nTriage Note: {vignette_grouped['chief_complaint']}\n{vignette_grouped['history_of_present_illness']}\nDiagnosis:{diagnosis}"

            # For any additional history input
            addit_hx_input = json.dumps(
                dict(
                    past_medical_history=vignette_grouped.get("past_medical_history"),
                    allergy=vignette_grouped.get("allergy"),
                    medication_history=vignette_grouped.get("medication_history"),
                    family_history=vignette_grouped.get("family_history"),
                    social_history=vignette_grouped.get("social_history"),
                    uncategorized=vignette_grouped.get("uncategorized"),
                )
            )
        except Exception as e:
            logger.warning("error dividing vignettes into structures")
            logger.exception(e)
            result = GenerationResult.model_validate(
                {
                    "status": "error processing vignette into structured groups",
                    "error_message": str(e),
                    "tokens": total_tokens,
                    "output": "",
                    "artifact": artifact,
                }
            )

            return result

        # Step 2: Parse sections into their respective structured format
        try:
            # Run tasks in parallel
            tasks = [
                self.async_model_invoke(
                    [
                        SystemMessage(HOPC_SPLIT_SYSTEM),
                        HumanMessage(HOPC_SPLIT_PROMPT.format(history=hopc)),
                    ]
                ),
                self.async_model_invoke(
                    [
                        SystemMessage(ADDIT_HISTORY_PARSE),
                        HumanMessage(addit_hx_input),
                    ]
                ),
                self.async_model_invoke(
                    [
                        SystemMessage(PHYSICAL_PARSE_PROMPT),
                        HumanMessage(vignette_grouped["physical_examination"]),
                    ]
                ),
                self.async_model_invoke(
                    [
                        SystemMessage(INVESTIGATION_PARSE_PROMPT),
                        HumanMessage(vignette_grouped["investigation_findings"]),
                    ]
                ),
                self.async_model_invoke(
                    [
                        SystemMessage(IMAGING_PARSE_PROMPT),
                        HumanMessage(vignette_grouped["investigation_findings"]),
                    ]
                ),
            ]

            results = await asyncio.gather(*tasks)

            hopc_response, hopc_tokens = results[0]
            addit_hx_response, addit_hx_tokens = results[1]
            physical_response, physical_tokens = results[2]
            ix_response, ix_tokens = results[3]
            imaging_response, imaging_tokens = results[4]

            total_tokens += (
                hopc_tokens
                + addit_hx_tokens
                + physical_tokens
                + ix_tokens
                + imaging_tokens
            )

            artifact["structured_history_raw"] = hopc_response.content
            artifact["addit_history_raw"] = addit_hx_response.content
            artifact["physical_raw"] = physical_response.content
            artifact["investigations_raw"] = ix_response.content
            artifact["imaging_raw"] = imaging_response.content

            try:
                structured_history = rapidjson.loads(
                    remove_json_markdown(hopc_response.content),
                    parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
                )

            except Exception as e:
                logger.exception(e)
                logger.info(hopc_response.content)
                raise e

            try:
                addit_hx = rapidjson.loads(
                    remove_json_markdown(addit_hx_response.content),
                    parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
                )
            except Exception as e:
                logger.exception(e)
                logger.info(addit_hx_response.content)
                raise e

            try:
                physical_exam = rapidjson.loads(
                    remove_json_markdown(physical_response.content),
                    parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
                )
            except Exception as e:
                logger.exception(e)
                logger.info(physical_response.content)
                raise e

            try:
                investigations = rapidjson.loads(
                    remove_json_markdown(ix_response.content),
                    parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
                )
            except Exception as e:
                logger.exception(e)
                logger.info(ix_response.content)
                raise e

            try:
                imaging = rapidjson.loads(
                    remove_json_markdown(imaging_response.content),
                    parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
                )
            except Exception as e:
                logger.exception(e)
                logger.info(imaging_response.content)
                raise e

        except Exception as e:
            logger.warning("Error parsing individual sections as .json")
            logger.exception(e)
            result = GenerationResult.model_validate(
                {
                    "status": "error parsing section .jsons",
                    "error_message": str(e),
                    "tokens": total_tokens,
                    "output": "",
                    "artifact": artifact,
                }
            )

            return result

        for k, v in addit_hx.items():
            structured_history[k] = v

        output = dict(
            structured_history=structured_history,
            history_freetext=vignette_grouped["history_of_present_illness"],
            physical=physical_exam,
            investigations=investigations,
            imaging=imaging,
        )

        result = GenerationResult.model_validate(
            {
                "status": "success",
                "error_message": "",
                "tokens": total_tokens,
                "output": output,
                "artifact": artifact,
            }
        )

        return result

    async def clean_structured_history(
        self, result: GenerationResult
    ) -> GenerationResult:

        SNOMED_THRESHOLD = 0.9

        structured_history = result.output.pop("structured_history")
        result.artifact["structured_history"] = deepcopy(structured_history)
        tokens_used = 0

        try:

            # Map all symptoms and associated symptoms to SNOMED specific terminology for standardization and retrieval
            freetext_terms = set([s["name"] for s in structured_history["symptoms"]])
            for s in structured_history["symptoms"]:
                if assoc := s.get("associated_symptoms"):
                    freetext_terms.update(set(assoc))

            # Create mapping input with embedding search results
            unmapped = []
            snomed_mapped = {}
            partial = []
            for freetext_term in freetext_terms:
                search_results = self.snomed_embeddings.search(freetext_term, limit=5)
                candidate_terms = [
                    t["text"] for t in search_results if t["score"] > SNOMED_THRESHOLD
                ]
                if not candidate_terms:
                    unmapped.append(freetext_term)
                elif len(candidate_terms) == 1:
                    snomed_mapped[freetext_term] = candidate_terms[0]
                else:
                    partial.append((freetext_term, candidate_terms))

            # We "rescue" these symptom findings by mapping them into specific keywords
            SNOMED_RESCUE_PROMPT = "Remove any references on location or laterality in this symptom, then convert this symptom to SNOMED-standardized terms:{symptom}. Return the single converted term only and nothing else"
            symptoms_to_map = [SNOMED_RESCUE_PROMPT.format(symptom=s) for s in unmapped]
            tasks = [self.async_model_invoke(m) for m in symptoms_to_map]

            rescue_results = await asyncio.gather(*tasks)
            tokens_used += sum(r[1] for r in rescue_results)

            rescued_ids = [
                r[0].content.replace("(finding)", "") for r in rescue_results
            ]

            # After rescue, search again
            for unmapped_term, rescued_term in zip(unmapped, rescued_ids):
                search_results = self.snomed_embeddings.search(rescued_term, limit=5)
                candidate_terms = [
                    t["text"] for t in search_results if t["score"] > SNOMED_THRESHOLD
                ]
                all_candidate_terms = [t["text"] for t in search_results]
                if len(candidate_terms) == 1:
                    snomed_mapped[unmapped_term] = candidate_terms[0]
                else:
                    partial.append((unmapped_term, all_candidate_terms))

            SNOMED_SELECTION_PROMPT = """Below is a tuple containing (original phrase, [candidate standardized phrases]) for medical data. Select the most appropriate candidate term that preserves the semantic meaning of the original phrase. Ignore any references on location or laterality in the original phrase. Be careful with negations, qualifiers, and contradictory terms. For example, if the original phrase is "non-productive cough" and the candidates are ["Productive cough", "Cough", "Chronic cough"], you should select "Cough" since "Productive cough" contradicts the original meaning, and "Chronic cough" adds additional information that does not reflect the original phrase. 
            If none of the supplied terms match the original phrase, return the original phrase. For example, if the original phrase is "fever", and the options are ["Pain", "Nausea", "Cough"], return "fever".
            Select the most appropriate standardized term for each medical phrase. Return single phrase only, corresponding to one of the candidate terms or the original phrase. 
            """

            # For multiple viable candidates, we use LLM to further map it to the best one
            tasks = [
                self.async_model_invoke(
                    [SystemMessage(SNOMED_SELECTION_PROMPT), HumanMessage(str(m))]
                )
                for m in partial
            ]
            selection_results = await asyncio.gather(*tasks)
            tokens_used += sum(r[1] for r in selection_results)

            selected_ids = [
                r[0].content.replace("(finding)", "") for r in selection_results
            ]

            for p, s in zip(partial, selected_ids):
                snomed_mapped[p[0]] = s

            # Then, we clean the chief complaint, and check each symptom on whether it is primary or not
            hx_clean_response, hx_clean_tokens = await self.async_model_invoke(
                [
                    SystemMessage(HX_CLEAN_PROMPT),
                    HumanMessage(
                        str(
                            dict(
                                chief_complaint=structured_history["chief_complaint"],
                                symptoms=list(snomed_mapped.keys()),
                            )
                        )
                    ),
                ]
            )

            hx_clean = rapidjson.loads(
                hx_clean_response.content,
                parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
            )
            tokens_used += hx_clean_tokens

            _symptoms = {}
            primary_mapping: Dict[str, Any] = hx_clean.get("symptom_mapping", {})

            chief_complaint = hx_clean.get("chief_complaint")

            # For each symptom, we map it to snomed standardized keys, and resolve any collisions (if any)
            for symptom in structured_history["symptoms"]:
                orig_name = symptom["name"]
                snomed_name = snomed_mapped.get(orig_name, orig_name)
                symptom["name"] = snomed_name
                symptom["primary"] = primary_mapping.get(orig_name, False)
                snomed_key = self.snomed_to_key(snomed_name)

                symptom["associated_symptoms"] = [
                    snomed_mapped.get(s, s)
                    for s in symptom.get("associated_symptoms", [])
                ]

                if snomed_key not in _symptoms.keys():
                    _symptoms[snomed_key] = symptom
                else:
                    existing_symptom = deepcopy(_symptoms[snomed_key])
                    logger.warning(
                        f"Key collision for symptom: {symptom}, {existing_symptom}"
                    )

                    # Handle presence (take the max - True has precedence over False)
                    if "present" in symptom or "present" in existing_symptom:
                        existing_symptom["present"] = max(
                            existing_symptom["present"], symptom["present"]
                        )

                    # Process all other attributes
                    for k, v in symptom.items():
                        if k in ["present", "system", "name"]:
                            continue  # Already handled above

                        elif k not in existing_symptom:
                            # If attribute only exists in new symptom, add it
                            existing_symptom[k] = v
                        elif v is not None:  # Only process if new value is not None
                            if isinstance(v, list):
                                # For list attributes, extend the existing list
                                if isinstance(existing_symptom[k], list):
                                    # Add only unique items
                                    existing_symptom[k].extend(
                                        [
                                            item
                                            for item in v
                                            if item not in existing_symptom[k]
                                        ]
                                    )
                            elif isinstance(v, str) and v.strip():
                                # For string attributes, concatenate with | if both exist and are non-empty
                                if (
                                    isinstance(existing_symptom[k], str)
                                    and existing_symptom[k].strip()
                                ):
                                    existing_symptom[k] = f"{existing_symptom[k]} | {v}"
                                else:
                                    existing_symptom[k] = v
                    _symptoms[snomed_key] = existing_symptom

            structured_history["chief_complaint"] = chief_complaint
            structured_history["symptoms"] = _symptoms

            result.output["demographics"] = structured_history.pop("demographic")
            result.output["history"] = structured_history

            result.tokens += tokens_used
            return result
        except Exception as e:
            logger.exception(e)
            result.status = "error at cleaning structured history"
            result.error_message = str(e)
            result.tokens += tokens_used

            return result

    async def validate_diagnosis(
        self,
        vignette: str,
        diagnosis: List[str],
        differentials: List[str],
        result: GenerationResult,
    ) -> GenerationResult:
        """Method to process / validate diagnosis items, and match freetext diagnosis items with keys from structured information"""

        try:
            clincase = ClinicalCase.model_validate(result.output)
            validation_input = DDX_VALIDATION_PROMPT.format(
                vignette=vignette,
                ddx=diagnosis,
                differentials=differentials,
                clin_dict=clincase.dict(),
            )
        except Exception as e:
            logger.exception(f"Error validating output as clinical case: {e}")
            result.status = "error at diagnosis validation: output validation"
            result.error_message = str(e)

            return result

        try:
            diagnosis_response, tokens_used = await self.async_model_invoke(
                [SystemMessage(DDX_VALIDATION_SYSTEM), HumanMessage(validation_input)]
            )
            result.tokens += tokens_used

            diagnosis_response_raw = remove_json_markdown(diagnosis_response.content)
            result.artifact["diagnosis_response_raw"] = diagnosis_response_raw

            try:
                possible_diagnosis = rapidjson.loads(
                    diagnosis_response_raw,
                    parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
                )
            except Exception as e:
                logger.exception(e)
                logger.info(diagnosis_response_raw)
                raise e

            result.output["diagnosis"] = possible_diagnosis.get(
                "confirmed_diagnoses", []
            )
            result.output["differentials"] = possible_diagnosis.get(
                "other_acceptable_diagnoses", []
            )

            result.artifact["validated_diagnoses"] = [
                d["name"] for d in result.output["diagnosis"]
            ]
            result.artifact["validated_differentials"] = [
                d["name"] for d in result.output["differentials"]
            ]

            return result

        except Exception as e:
            logger.exception(f"Error getting diagnosis return: {e}")
            result.status = "error at diagnosis validation: diagnosis return"
            result.error_message = str(e)

            return result

    async def generate_case(
        self,
        vignette: str,
        generate_diagnosis=True,
        diagnosis: Union[str, List[str]] = [],
        differentials: List[str] = [],
    ) -> Dict[str, Any]:
        """
        Main entry point to generate a structured clinical case from an unstructured vignette.

        Args:
            vignette: Unstructured clinical vignette
            diagnosis: Optional diagnosis

        Returns:
            Dictionary with processing results and structured case
        """
        # Convert into structured data close-enough to our own format
        result = await self.process_vignette(vignette, diagnosis)
        if result.status != "success":
            logger.warning(f"Error at process vignette: {result.error_message}")
            return result.model_dump()

        result = await self.clean_structured_history(result)

        if result.status != "success":
            logger.warning(f"Error at clean history: {result.error_message}")
            return result.model_dump()

        ddx = None
        if generate_diagnosis:
            logger.info("Parsing diagnosis from vignette")
            ddx, result = await self.parse_diagnosis(result, vignette)

        if ddx:
            diagnosis = ddx.get("primary_diagnoses", [])
            differentials = ddx.get("differential_diagnoses", [])
            clinical_notes = ddx.get("clinical_notes", "")

            result.artifact["diagnosis_parsed"] = diagnosis
            result.artifact["differentials_parsed"] = differentials
            result.artifact["ddx_clinical_notes"] = clinical_notes

            if clinical_notes:
                differentials.append(clinical_notes)
        else:
            diagnosis = result.artifact["vignette_grouped"]["diagnosis_freetext"]
            if isinstance(diagnosis, str):
                diagnosis = [diagnosis]
            logger.warning(
                f"No diagnosis provided in input. Inferring diagnosis from vignette: {diagnosis}"
            )

        result = await self.validate_diagnosis(
            vignette, diagnosis, differentials, result
        )

        if result.status != "success":
            logger.warning(f"Error at validate diagnosis: {result.error_message}")
            return result.model_dump()

        # Save results if output file is specified
        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(result.model_dump()) + "\n")

        return result.model_dump()


class MedQACaseGenerator(CaseGenerator):

    def __init__(self, model: BaseChatModel, output_file: str = ""):
        self.model = model
        self.output_file = output_file

    async def expand_medqa(self, medqa_prompt: str):
        """Expand a clinical vignette from the MedQA dataset"""
        tokens = 0
        try:
            tasks = [
                self.async_model_invoke(
                    [
                        SystemMessage(MEDQA_HISTORY_EXPAND_PROMPT),
                        HumanMessage(medqa_prompt),
                    ]
                ),
                self.async_model_invoke(
                    [
                        SystemMessage(MEDQA_PHYSICAL_EXPAND_PROMPT),
                        HumanMessage(medqa_prompt),
                    ]
                ),
                self.async_model_invoke(
                    [
                        SystemMessage(MEDQA_INVESTIGATION_EXPAND_PROMPT),
                        HumanMessage(medqa_prompt),
                    ]
                ),
            ]

            results = await asyncio.gather(*tasks)

            expanded_hx, hx_tokens = results[0]
            expanded_physical, physical_tokens = results[1]
            expanded_ix, ix_tokens = results[2]

            tokens += hx_tokens + physical_tokens + ix_tokens

            return (
                None,
                expanded_hx.content,
                expanded_physical.content,
                expanded_ix.content,
                tokens,
            )

        except Exception as e:
            return e, None, None, None, tokens

    async def parse_case(
        self,
        history_input,
        physical_input,
        investigations_input,
        diagnosis: List[str] = [],
    ):
        tokens = 0
        artifact = dict(
            history_input=history_input,
            physical_input=physical_input,
            investigations_input=investigations_input,
            imaging=None,
            diagnosis=diagnosis,
        )
        try:
            tasks = [
                self.async_model_invoke([HISTORY_PARSE_AUGMENTED, history_input]),
                self.async_model_invoke([PHYSICAL_PARSE_AUG_PROMPT, physical_input]),
                self.async_model_invoke(
                    [MEDQA_INVESTIGATION_PARSE_PROMPT, investigations_input]
                ),
            ]

            results = await asyncio.gather(*tasks)

            history_response, history_tokens = results[0]
            physical_response, physical_tokens = results[1]
            investigations_response, investigations_tokens = results[2]

            tokens += history_tokens + physical_tokens + investigations_tokens

            _history_demographics_raw = remove_json_markdown(history_response.content)
            artifact["_history_demographics"] = _history_demographics_raw
            _history_demographics = rapidjson.loads(
                _history_demographics_raw,
                parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
            )
            demographics_parsed = Demographics.model_validate(
                _history_demographics["demographics"]
            )

            history_semi_parsed = _history_demographics["history"]
            symptoms_free_text = history_semi_parsed["symptoms_freetext"]

            tasks = [
                # Parse symptoms again. This one is long
                self.async_model_invoke([SYMPTOMS_PARSE_AUGMENTED, symptoms_free_text]),
                # Imaging is just expanded and parsed in one go
                self.async_model_invoke(
                    [
                        MEDQA_IMAGING_PARSE_EXPAND_PROMPT,
                        str(history_input)
                        + str(physical_input)
                        + str(investigations_input)
                        + "Diagnosis: "
                        + str(diagnosis),
                    ]
                ),
            ]

            results = await asyncio.gather(*tasks)

            symptoms_response, symptom_tokens = results[0]
            imaging_response, imaging_tokens = results[1]

            tokens += symptom_tokens + imaging_tokens

            _symptoms_raw = remove_json_markdown(symptoms_response.content)
            artifact["_symptoms_raw"] = _symptoms_raw
            history_semi_parsed["symptoms"] = rapidjson.loads(
                _symptoms_raw,
                parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
            )
            history_parsed = History.model_validate(history_semi_parsed)

            _physical_raw = remove_json_markdown(physical_response.content)
            artifact["_physical"] = _physical_raw
            _physical = rapidjson.loads(
                _physical_raw,
                parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
            )
            physical_parsed = PhysicalExamination.model_validate(_physical)

            _investigations_raw = remove_json_markdown(investigations_response.content)
            artifact["_investigations"] = _investigations_raw
            _investigations = rapidjson.loads(
                _investigations_raw,
                parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
            )
            investigations_parsed = Investigations.model_validate(_investigations)

            imaging_parsed = rapidjson.loads(
                remove_json_markdown(imaging_response.content),
                parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
            )
            artifact["imaging"] = imaging_parsed

        except Exception as e:
            return {
                "status": "error at parsing",
                "error_message": str(e),
                "tokens": tokens,
                "output": None,
                "artifact": artifact,
            }

        return {
            "status": "success",
            "error_message": None,
            "tokens": tokens,
            "output": ClinicalCase(
                demographics=demographics_parsed,
                history=history_parsed,
                history_freetext=history_input,
                physical=physical_parsed,
                investigations=investigations_parsed,
                imaging=imaging_parsed,
                diagnosis=diagnosis,
            ).model_dump(),
            "artifact": artifact,
        }

    async def generate_medqa_case(self, medqa_prompt: str, diagnosis: str = ""):

        # Kinda ugly to be doing it here but this will do for now
        error_msg, history, physical, investigations, _tokens = await self.expand_medqa(
            medqa_prompt
        )

        if error_msg:
            result = {
                "status": "error at expansion",
                "error_message": error_msg,
                "tokens": _tokens,
                "output": None,
            }
        else:
            result = await self.parse_case(history, physical, investigations, diagnosis)
            result["tokens"] += _tokens

        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(result))

        return result
