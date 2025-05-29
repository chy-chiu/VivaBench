from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import txtai
from loguru import logger
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from vivabench.ontology.schema import Differential


class AgentDiagnosis(BaseModel):

    condition: str
    icd_10_name: str
    icd_10: str
    confidence: float


class DiagnosisMatch(BaseModel):
    match_type: Literal["exact", "approximate", "none"] = "none"
    matched_ground_truth: Optional[Differential] = None
    model_output_idx: int
    confidence: float
    diagnosis_type: str
    similarity_score: Optional[float] = None


class EvaluationMetrics:
    def __init__(
        self,
        semantic_similarity_threshold=0.8,
        icd_embedding_path="./medical/icd_embeddings",
        icd_mapping_path="./medical/d_icd_diagnoses.csv",
        sentence_transformer_model="all-mpnet-base-v2",
    ):

        if icd_embedding_path:
            print("Using preloaded embeddings for icd-10 mapping")
            self.icd10_embeddings = txtai.Embeddings(
                path="neuml/pubmedbert-base-embeddings", content=True
            )
            self.icd10_embeddings.load(icd_embedding_path)
        else:
            raise ValueError("need ICD-10 embeddings path")

        # Load sentence transformer model
        self.embedding_model = SentenceTransformer(sentence_transformer_model)

        # Load ICD-10 mappings
        icd10 = pd.read_csv(icd_mapping_path).query("icd_version==10")
        self.icd10_codes = icd10.icd_code.to_list()
        self.icd10_mapping = icd10.set_index("icd_code").long_title.to_dict()

        # Set default threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold

        # Initialize other attributes with empty values
        self.gt_diagnosis: List[Differential] = []
        self.accepted_differentials: List[Differential] = []
        self.provisional_diagnosis: List[AgentDiagnosis] = []
        self.final_diagnosis: List[AgentDiagnosis] = []
        self.full_info_diagnosis: List[AgentDiagnosis] = []
        self.matched_keys = set()
        self.unmatched_case_keys = set()
        self.unmatched_request_keys = set()

        # Initialize caches
        self.semantic_cache = {}
        self.condition_embeddings = {}

        # Initialize metrics containers
        self._init_metrics_containers()

    def validate_diagnosis(self, ddx: Differential):

        condition_name = ddx.name
        orig_icd_10 = ddx.icd_10.replace(".", "")
        orig_icd_10_name = (
            ddx.icd_10_name if ddx.icd_10_name else self.icd10_mapping.get(orig_icd_10)
        )

        # If error in obtaining icd 10 name, we move up one level
        if not orig_icd_10_name:
            orig_icd_10 = orig_icd_10[:3]
            orig_icd_10_name = self.icd10_mapping.get(orig_icd_10)
            if not orig_icd_10_name:
                logger.warning(
                    f"Unable to match despite going up one level higher: {orig_icd_10}, {condition_name}"
                )
                return None

        if not self._is_semantic_match(orig_icd_10_name, condition_name, threshold=0.8):
            matched_icd10_code, matched_icd10_desc = self.validate_icd10_code(
                orig_icd_10, condition_name
            )

            # If both are shortened, we just keep the original one
            if len(matched_icd10_code) == 3 and len(orig_icd_10) == 3:

                ddx.icd_10_name = orig_icd_10_name
                ddx.icd_10 = orig_icd_10

            else:
                ddx.icd_10_name = matched_icd10_desc
                ddx.icd_10 = matched_icd10_code

        return ddx

    def validate_icd10_code(self, icd10_code, icd10_desc):
        APPROX_THRESHOLD = 0.7
        EXACT_THRESHOLD = 0.99

        # COVID-19 override
        if "COVID-19" in icd10_desc:
            matched_icd10_code = "U071"
            matched_icd10_desc = "COVID-19"
            return matched_icd10_code, matched_icd10_desc

        _icd10_code = icd10_code.replace(".", "")

        icd_description_matches = self.icd10_embeddings.search(icd10_desc)

        matched_icd10_desc = ""
        matched_icd10_code = ""
        for m in icd_description_matches:
            if m["score"] > EXACT_THRESHOLD:
                matched_icd10_desc = m["text"]
                matched_icd10_code = self.icd10_codes[int(m["id"])]

        _icd10_desc_from_code = self.icd10_mapping.get(_icd10_code)
        for m in icd_description_matches:
            if _icd10_desc_from_code == m["text"]:
                matched_icd10_desc = m["text"]
                matched_icd10_code = _icd10_code

        else:
            candidates = [m for m in icd_description_matches if m["score"]]
            c = [_c for _c in candidates if _c["score"] > APPROX_THRESHOLD]
            if c:
                m = c[0]
                matched_icd10_desc = m["text"]
                matched_icd10_code = self.icd10_codes[int(m["id"])]

        if not matched_icd10_code or not matched_icd10_desc:
            matched_icd10_code = icd10_code[:3]
            matched_icd10_desc = self.icd10_mapping.get(matched_icd10_code)

            if not matched_icd10_desc:
                logger.warning(
                    f"Unable to match despite going up one level higher! {matched_icd10_code}, {icd10_desc}"
                )
                return icd10_code, icd10_desc

        return matched_icd10_code, matched_icd10_desc

    def _init_metrics_containers(self):
        """Initialize all metrics containers with empty values"""
        # Results storage
        self.matches = {"final": [], "provisional": [], "full_info": []}

        # Accuracy metrics
        self.top_k_exact_accuracies = {"final": {}, "provisional": {}, "full_info": {}}
        self.top_k_approx_accuracies = {"final": {}, "provisional": {}, "full_info": {}}

        # Confidence scores
        self.confidence_scores = {"final": 0.0, "provisional": 0.0, "full_info": 0.0}

        # Key relevance metrics
        self.key_relevance_metrics = {}

        # Diagnostic change metrics
        self.diagnostic_change_metrics = {}

        #
        self.confidence_values = {
            "confidence_value_final": 0.0,
            "confidence_value_provisional": 0.0,
            "confidence_value_full_info": 0.0,
            "confidence_value_final_exact": 0.0,
            "confidence_value_provisional_exact": 0.0,
            "confidence_value_full_info_exact": 0.0,
            "confidence_value_final_approx_exact": 0.0,
            "confidence_value_provisional_approx_exact": 0.0,
            "confidence_value_full_info_approx_exact": 0.0,
            "confidence_value_final_unmatched": 0.0,
            "confidence_value_provisional": 0.0,
            "confidence_value_full_info": 0.0,
        }

    def load_results(
        self,
        gt_diagnosis: List[Dict[str, Any]],
        gt_differentials: List[Dict[str, Any]],
        final_diagnosis: List[Dict[str, Any]],
        provisional_diagnosis: List[Dict[str, Any]],
        full_info_diagnosis: List[Dict[str, Any]],
        matched_keys: Iterable[str],
        unmatched_request_keys: Iterable[str],
        unmatched_case_keys: Iterable[str],
    ):
        """Load results data and reset all metrics"""
        # Reset all metrics
        self._init_metrics_containers()

        # Reset caches if needed for new case
        self.condition_embeddings = {}

        # Load new data
        gt_diagnosis: List[Differential] = [
            Differential.model_validate(d) for d in gt_diagnosis
        ]
        gt_differentials: List[Differential] = [
            Differential.model_validate(d) for d in gt_differentials
        ]

        # self.gt_diagnosis = gt_diagnosis
        # self.accepted_differentials= gt_differentials

        _gts = []
        for gt in gt_diagnosis:
            if gt := self.validate_diagnosis(gt):
                _gts.append(gt)
        self.gt_diagnosis = _gts

        _gts = []
        for gt in gt_differentials:
            if gt := self.validate_diagnosis(gt):
                _gts.append(gt)
        self.accepted_differentials = _gts

        self.final_diagnosis = [
            AgentDiagnosis.model_validate(d) for d in final_diagnosis
        ]
        self.provisional_diagnosis = [
            AgentDiagnosis.model_validate(d) for d in provisional_diagnosis
        ]
        self.full_info_diagnosis = [
            AgentDiagnosis.model_validate(d) for d in full_info_diagnosis
        ]

        self.matched_keys = set(matched_keys)
        self.unmatched_request_keys = set(unmatched_request_keys)
        self.unmatched_case_keys = set(unmatched_case_keys)

    def compute_all_metrics(self):
        """Compute all metrics for all available diagnosis types"""
        # First find matches for each diagnosis type
        for diag_type in ["final", "provisional", "full_info"]:
            if diag_type == "final" and self.final_diagnosis:
                self.find_matches(diag_type)
            elif diag_type == "provisional" and self.provisional_diagnosis:
                self.find_matches(diag_type)
            elif diag_type == "full_info" and self.full_info_diagnosis:
                self.find_matches(diag_type)

        # Then compute metrics for each diagnosis type
        for diag_type in ["final", "provisional", "full_info"]:
            if self.matches.get(diag_type):
                self.compute_top_k_accuracy(diag_type)
                self.compute_confidence_score(diag_type)

        # hist_phys=True, investigations=True, from_matched_gt=True):
        for prefix, config in zip(
            ["hp_matched", "hp_all", "ix_matched", "ix_all"],
            [
                (True, False, True),
                (True, False, False),
                (False, True, True),
                (False, True, False),
            ],
        ):
            hp, ix, m = config
            key_metrics = self.compute_key_relevance(
                hist_phys=hp, investigations=ix, from_matched_gt=m
            )

            key_metrics = {f"{prefix}_{k}": v for k, v in key_metrics.items()}

            self.key_relevance_metrics[prefix] = key_metrics

        # Compute diagnostic changes
        if self.provisional_diagnosis and self.final_diagnosis:
            self.compute_diagnostic_changes()

        self._compute_confidence_values()

        return self.summarize_results()

    def compute_embeddings_for_all_conditions(self):
        """Compute embeddings for all condition names in the dataset"""
        # Collect all unique condition names
        all_conditions = set()

        # From ground truth
        for diag in self.gt_diagnosis:
            all_conditions.add(diag.name)
            if hasattr(diag, "icd_10_name") and diag.icd_10_name:
                all_conditions.add(diag.icd_10_name)

        # From accepted differentials
        if self.accepted_differentials:
            for diag in self.accepted_differentials:
                all_conditions.add(diag.name)
                if hasattr(diag, "icd_10_name") and diag.icd_10_name:
                    all_conditions.add(diag.icd_10_name)

        # From model outputs
        for diag in self.final_diagnosis:
            all_conditions.add(diag.condition)
            all_conditions.add(diag.icd_10_name)

        if self.provisional_diagnosis:
            for diag in self.provisional_diagnosis:
                all_conditions.add(diag.condition)
                all_conditions.add(diag.icd_10_name)

        if self.full_info_diagnosis:
            for diag in self.full_info_diagnosis:
                all_conditions.add(diag.condition)
                all_conditions.add(diag.icd_10_name)

        # Remove any None/empty values
        all_conditions = [c for c in all_conditions if c]

        # Compute embeddings
        try:
            condition_texts = list(all_conditions)
            embeddings = self.embedding_model.encode(
                condition_texts, convert_to_tensor=True
            )

            # Store in cache
            for i, condition in enumerate(condition_texts):
                self.condition_embeddings[condition] = embeddings[i]

            # print(f"Computed embeddings for {len(condition_texts)} conditions")
        except Exception as e:
            print(f"Error computing embeddings: {str(e)}")

    def _get_embedding(self, text):
        """Get embedding for a text, computing it if necessary"""
        if not text or not self.embedding_model:
            return None

        if text not in self.condition_embeddings:
            try:
                embedding = self.embedding_model.encode(text, convert_to_tensor=True)
                self.condition_embeddings[text] = embedding
                return embedding
            except Exception as e:
                logger.exception(e)
                return None

        return self.condition_embeddings[text]

    def _compute_confidence_values(self):
        self.confidence_values = {
            "confidence_value_final": 0.0,
            "confidence_value_provisional": 0.0,
            "confidence_value_full_info": 0.0,
            "confidence_value_final_exact": 0.0,
            "confidence_value_provisional_exact": 0.0,
            "confidence_value_full_info_exact": 0.0,
            "confidence_value_final_approx_exact": 0.0,
            "confidence_value_provisional_approx_exact": 0.0,
            "confidence_value_full_info_approx_exact": 0.0,
            "confidence_value_final_unmatched": 0.0,
            "confidence_value_provisional": 0.0,
            "confidence_value_full_info": 0.0,
        }

        ref_map = {
            "full_info": self.full_info_diagnosis,
            "provisional": self.provisional_diagnosis,
            "final": self.final_diagnosis,
        }

        for phase in ["full_info", "provisional", "final"]:

            exact_match_confidence = []
            approx_exact_match_confidence = []
            unmatched_confidence = []

            for match in self.matches[phase]:
                model_output_idx = match.model_output_idx
                ddx_confidence = float(ref_map[phase][model_output_idx].confidence)

                if match.match_type == "exact":
                    exact_match_confidence.append(ddx_confidence)
                    approx_exact_match_confidence.append(ddx_confidence)
                elif match.match_type == "approximate":
                    approx_exact_match_confidence.append(ddx_confidence)
                if match.match_type == "none":
                    unmatched_confidence.append(ddx_confidence)

            all_ddx_confidence = (
                exact_match_confidence
                + approx_exact_match_confidence
                + unmatched_confidence
            )
            if all_ddx_confidence:
                self.confidence_values[f"confidence_value_{phase}"] = np.mean(
                    all_ddx_confidence
                )
            if exact_match_confidence:
                self.confidence_values[f"confidence_value_{phase}_exact"] = np.mean(
                    exact_match_confidence
                )
            if approx_exact_match_confidence:
                self.confidence_values[f"confidence_value_{phase}_approx_exact"] = (
                    np.mean(approx_exact_match_confidence)
                )
            if unmatched_confidence:
                self.confidence_values[f"confidence_value_{phase}_unmatched"] = np.mean(
                    unmatched_confidence
                )

        return

    def _compute_similarity_matrix(
        self, texts1: List[str], texts2: Optional[List[str]] = None
    ):
        """Compute cosine similarity matrix between two lists of texts"""
        if not self.embedding_model:
            return None

        if texts2 is None:
            texts2 = texts1

        # Get embeddings
        embeddings1 = [self._get_embedding(text) for text in texts1]
        if any(e is None for e in embeddings1):
            return None

        if texts1 is texts2:
            embeddings2 = embeddings1
        else:
            embeddings2 = [self._get_embedding(text) for text in texts2]
            if any(e is None for e in embeddings2):
                return None

        # Stack embeddings
        stacked1 = torch.stack(embeddings1)
        stacked2 = torch.stack(embeddings2)

        # Compute cosine similarity
        similarity = F.cosine_similarity(
            stacked1.unsqueeze(1), stacked2.unsqueeze(0), dim=2
        )

        return similarity

    def _icd10_is_exact_match(self, model_icd, gt_icd):
        """Check if ICD-10 codes match exactly at the appropriate level"""
        # Clean codes
        model_code = model_icd.replace(".", "")
        gt_code = gt_icd.replace(".", "")

        # Clip to first 3 levels (first one is letter)
        if len(model_code) >= 4:
            model_code = model_code[:4]
        if len(gt_code) >= 4:
            gt_code = gt_code[:4]

        # If ground truth has fewer digits, check prefix match
        if len(gt_code) < len(model_code):
            return model_code.startswith(gt_code)
        # If model code has equal digits, must match up to exact
        elif len(gt_code) == len(model_code):
            return gt_code == model_code
        else:
            return False

    def _icd10_is_approximate_match(self, model_icd, gt_icd):
        """Check if ICD-10 codes match approximately"""
        model_code = model_icd.replace(".", "")
        gt_code = gt_icd.replace(".", "")

        # If they share first 3 chars but aren't exact matches
        if len(model_code) >= 3 and len(gt_code) >= 3:
            return model_code[:3] == gt_code[:3] and not self._icd10_is_exact_match(
                model_icd, gt_icd
            )

        return False

    def _is_semantic_match(self, model_string, gt_string, threshold=None):
        """Check for semantic similarity using embeddings and cosine similarity"""
        # Fall back to string matching if no embedding model
        if not self.embedding_model:
            return self._fallback_semantic_match(model_string, gt_string)

        threshold = threshold or self.semantic_similarity_threshold

        # Get embeddings
        model_emb = self._get_embedding(model_string)
        gt_emb = self._get_embedding(gt_string)

        if model_emb is None or gt_emb is None:
            return self._fallback_semantic_match(model_string, gt_string)

        # Compute similarity
        similarity = F.cosine_similarity(
            model_emb.unsqueeze(0), gt_emb.unsqueeze(0), dim=1
        ).item()

        # Return similarity score if above threshold
        return similarity >= threshold

    def _fallback_semantic_match(self, model_string, gt_string):
        """Fallback semantic matching when embeddings aren't available"""
        if not model_string or not gt_string:
            return False

        model_lower = model_string.lower()
        gt_lower = gt_string.lower()

        # Simple Jaccard similarity on words
        model_words = set(model_lower.split())
        gt_words = set(gt_lower.split())

        if not model_words or not gt_words:
            return False

        intersection = model_words.intersection(gt_words)
        union = model_words.union(gt_words)

        jaccard = len(intersection) / len(union)
        return jaccard >= 0.5  # Threshold for Jaccard similarity

    def find_matches(self, diagnosis_type: str = "final"):
        """Match model outputs to ground truth diagnoses for a specific diagnosis type"""
        # Compute embeddings for all conditions if we have an embedding model
        if not self.condition_embeddings:
            self.compute_embeddings_for_all_conditions()

        # Get the correct diagnosis list
        if diagnosis_type == "final":
            diagnoses = self.final_diagnosis
        elif diagnosis_type == "provisional":
            diagnoses = self.provisional_diagnosis
        elif diagnosis_type == "full_info":
            diagnoses = self.full_info_diagnosis
        else:
            raise ValueError(f"Unknown diagnosis type: {diagnosis_type}")

        if not diagnoses:
            self.matches[diagnosis_type] = []
            return

        matches = []
        # Normalize confidence scores
        total_confidence = sum(d.confidence for d in diagnoses)

        for idx, diagnosis in enumerate(diagnoses):
            normalized_conf = (
                diagnosis.confidence / total_confidence if total_confidence > 0 else 0
            )

            # Try to find a match in ground truth
            match = DiagnosisMatch(
                model_output_idx=idx,
                confidence=normalized_conf,
                diagnosis_type=diagnosis_type,
            )

            # Check all ground truth diagnoses for a match
            all_gt: List[Differential] = list(self.gt_diagnosis)
            if self.accepted_differentials:
                all_gt.extend(self.accepted_differentials)

            # First try to find an exact / approximate match in ICD-10 codes
            for gt_idx, gt_diagnosis in enumerate(all_gt):
                if self._icd10_is_exact_match(diagnosis.icd_10, gt_diagnosis.icd_10):
                    match.match_type = "exact"
                    match.matched_ground_truth = all_gt[gt_idx]
                    match.similarity_score = 1.0  # Perfect match
                    break

                # If it's an accepted differential, consider it approximate even if exact ICD match
                if gt_idx >= len(self.gt_diagnosis):
                    match.match_type = "approximate"
                    match.matched_ground_truth = all_gt[gt_idx]
                    match.similarity_score = 0.9  # High but not perfect
                    break

            # If no exact match, look for approximate matches
            if match.match_type == "none":
                best_similarity = 0.0
                best_gt_idx = None

                for gt_idx, gt_diagnosis in enumerate(all_gt):
                    # Check ICD-10 approximate match
                    if self._icd10_is_approximate_match(
                        diagnosis.icd_10, gt_diagnosis.icd_10
                    ):
                        similarity = 0.8  # Good approximate match
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_gt_idx = gt_idx

                    # Check semantic similarity
                    # Compare condition names
                    if self._is_semantic_match(diagnosis.condition, gt_diagnosis.name):
                        # Get actual similarity score
                        model_emb = self._get_embedding(diagnosis.condition)
                        gt_emb = self._get_embedding(gt_diagnosis.name)
                        if model_emb is not None and gt_emb is not None:
                            similarity = F.cosine_similarity(
                                model_emb.unsqueeze(0), gt_emb.unsqueeze(0), dim=1
                            ).item()
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_gt_idx = gt_idx

                    # Also compare ICD-10 names if available
                    if hasattr(diagnosis, "icd_10_name") and self._is_semantic_match(
                        diagnosis.icd_10_name, gt_diagnosis.name
                    ):
                        model_emb = self._get_embedding(diagnosis.icd_10_name)
                        gt_emb = self._get_embedding(gt_diagnosis.name)
                        if model_emb is not None and gt_emb is not None:
                            similarity = F.cosine_similarity(
                                model_emb.unsqueeze(0), gt_emb.unsqueeze(0), dim=1
                            ).item()
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_gt_idx = gt_idx

                # If we found a good match, use it
                if (
                    best_gt_idx is not None
                    and best_similarity >= self.semantic_similarity_threshold
                ):

                    # print(
                    #     f"Model diagnosis: {diagnosis.icd_10_name} | GT: {all_gt[best_gt_idx].icd_10_name} | Similarity: {best_similarity}"
                    # )
                    if best_similarity == 1:
                        match.match_type = "exact"
                    else:
                        match.match_type = "approximate"

                    match.matched_ground_truth = all_gt[best_gt_idx]
                    match.similarity_score = best_similarity

            matches.append(match)

        self.matches[diagnosis_type] = matches

    def compute_top_k_accuracy(self, diagnosis_type: str = "final"):
        """Compute top-k exact and approximate accuracy metrics separately"""
        matches = self.matches.get(diagnosis_type, [])
        if not matches:
            self.top_k_exact_accuracies[diagnosis_type] = {}
            self.top_k_approx_accuracies[diagnosis_type] = {}
            return

        # Get the correct diagnosis list
        if diagnosis_type == "final":
            diagnoses = self.final_diagnosis
        elif diagnosis_type == "provisional":
            diagnoses = self.provisional_diagnosis
        elif diagnosis_type == "full_info":
            diagnoses = self.full_info_diagnosis

        for k in range(1, min(6, len(diagnoses) + 1)):
            # Get matches in top-k predictions
            top_k_matches = matches[:k]

            # For exact matches: any match is sufficient (binary outcome)
            has_exact_match = any(m.match_type == "exact" for m in top_k_matches)
            self.top_k_exact_accuracies[diagnosis_type][k] = (
                1.0 if has_exact_match else 0.0
            )

            # For approximate matches: any exact OR approximate match is sufficient
            has_approx_match = any(
                m.match_type in ["exact", "approximate"] for m in top_k_matches
            )
            self.top_k_approx_accuracies[diagnosis_type][k] = (
                1.0 if has_approx_match else 0.0
            )

    def compute_confidence_score(self, diagnosis_type: str = "final"):
        """Compute confidence-weighted score for a specific diagnosis type"""
        matches = self.matches.get(diagnosis_type, [])
        if not matches:
            self.confidence_scores[diagnosis_type] = 0.0
            return

        exact_match_conf = sum(m.confidence for m in matches if m.match_type == "exact")
        approx_match_conf = sum(
            m.confidence for m in matches if m.match_type == "approximate"
        )
        unmatched_conf = sum(m.confidence for m in matches if m.match_type == "none")

        # Final confidence score
        self.confidence_scores[diagnosis_type] = (
            exact_match_conf + approx_match_conf - unmatched_conf
        )

    def compute_key_relevance(
        self, hist_phys=True, investigations=True, from_matched_gt=True
    ):
        """Analyze relevance of keys ordered by the model for a specific diagnosis type"""
        _relevant_keys = set()

        if from_matched_gt:

            for match in self.matches["provisional"]:
                if matched_gt := match.matched_ground_truth:
                    _relevant_keys.update(set(matched_gt.relevant_keys))
            for match in self.matches["final"]:
                if matched_gt := match.matched_ground_truth:
                    _relevant_keys.update(set(matched_gt.relevant_keys))

        else:
            # Get all relevant keys from all ground truth diagnoses
            for gt_diagnosis in self.gt_diagnosis:
                _relevant_keys.update(gt_diagnosis.relevant_keys)

            if self.accepted_differentials:
                for diff in self.accepted_differentials:
                    _relevant_keys.update(diff.relevant_keys)

        _matched_keys = self.matched_keys
        matched_keys = set()
        all_relevant_keys = set()

        if hist_phys:
            matched_keys.update(
                set(
                    s
                    for s in _matched_keys
                    if s.startswith("history") or s.startswith("physical")
                )
            )
            all_relevant_keys.update(
                set(
                    s
                    for s in _relevant_keys
                    if s.startswith("history") or s.startswith("physical")
                )
            )

        if investigations:
            matched_keys.update(
                set(
                    s
                    for s in _matched_keys
                    if s.startswith("investigation") or s.startswith("imaging")
                )
            )
            all_relevant_keys.update(
                set(
                    s
                    for s in _relevant_keys
                    if s.startswith("investigation") or s.startswith("imaging")
                )
            )

        # Calculate overlap metrics
        relevant_ordered = matched_keys.intersection(all_relevant_keys)

        if len(matched_keys) > 0:
            precision = len(relevant_ordered) / len(matched_keys)
        else:
            precision = 0.0

        if len(all_relevant_keys) > 0:
            recall = len(relevant_ordered) / len(all_relevant_keys)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "relevant_ordered_count": len(relevant_ordered),
            "total_ordered_count": len(matched_keys),
            "total_relevant_count": len(all_relevant_keys),
        }

    def compute_diagnostic_changes(self):
        """Analyze changes between provisional and final diagnoses including accuracy changes"""
        if not self.provisional_diagnosis or not self.final_diagnosis:
            return

        # Basic diagnostic changes
        prov_icd_codes = {d.icd_10 for d in self.provisional_diagnosis}
        final_icd_codes = {d.icd_10 for d in self.final_diagnosis}

        added = final_icd_codes - prov_icd_codes
        removed = prov_icd_codes - final_icd_codes
        maintained = prov_icd_codes.intersection(final_icd_codes)

        # Confidence shifts for maintained diagnoses
        confidence_shifts = {}
        for icd in maintained:
            prov_conf = next(
                (d.confidence for d in self.provisional_diagnosis if d.icd_10 == icd), 0
            )
            final_conf = next(
                (d.confidence for d in self.final_diagnosis if d.icd_10 == icd), 0
            )
            confidence_shifts[icd] = final_conf - prov_conf

        # Accuracy changes
        prov_exact_top1 = self.top_k_exact_accuracies.get("provisional", {}).get(1, 0.0)
        final_exact_top1 = self.top_k_exact_accuracies.get("final", {}).get(1, 0.0)

        prov_approx_top1 = self.top_k_approx_accuracies.get("provisional", {}).get(
            1, 0.0
        )
        final_approx_top1 = self.top_k_approx_accuracies.get("final", {}).get(1, 0.0)

        # Confidence score changes
        prov_conf_score = self.confidence_scores.get("provisional", 0.0)
        final_conf_score = self.confidence_scores.get("final", 0.0)

        self.diagnostic_change_metrics = {
            "diagnoses_added": len(added),
            "diagnoses_removed": len(removed),
            "diagnoses_maintained": len(maintained),
            "confidence_shifts": (
                np.mean(list(confidence_shifts.values())) if confidence_shifts else 0.0
            ),
            "total_change_magnitude": sum(
                abs(shift) for shift in confidence_shifts.values()
            ),
            "exact_accuracy_change": final_exact_top1 - prov_exact_top1,
            "approx_accuracy_change": final_approx_top1 - prov_approx_top1,
            "confidence_score_change": final_conf_score - prov_conf_score,
        }

    def summarize_results(self) -> Dict:
        """Produce a single-row summary of all key metrics"""
        summary = {}

        # Top-k accuracies (k=1 to k=5) for each diagnosis type
        for diag_type in ["final", "provisional", "full_info"]:
            # Get available k values for this diagnosis type
            exact_k_values = sorted(
                self.top_k_exact_accuracies.get(diag_type, {}).keys()
            )
            approx_k_values = sorted(
                self.top_k_approx_accuracies.get(diag_type, {}).keys()
            )

            # Add top-k metrics for k=1 to k=5
            for k in range(1, 6):
                # Find best available k value
                exact_k = max([i for i in exact_k_values if i <= k] or [0])
                approx_k = max([i for i in approx_k_values if i <= k] or [0])

                # Get accuracies for the best available k
                if exact_k > 0:
                    summary[f"{diag_type}_top{k}_exact"] = self.top_k_exact_accuracies[
                        diag_type
                    ][exact_k]
                else:
                    summary[f"{diag_type}_top{k}_exact"] = 0.0

                if approx_k > 0:
                    summary[f"{diag_type}_top{k}_approx"] = (
                        self.top_k_approx_accuracies[diag_type][approx_k]
                    )
                else:
                    summary[f"{diag_type}_top{k}_approx"] = 0.0

            # Add confidence score
            summary[f"{diag_type}_confidence_score"] = self.confidence_scores.get(
                diag_type, 0.0
            )

        # Key relevance metrics
        for v in self.key_relevance_metrics.values():
            summary.update(v)

        # Diagnostic changes
        if self.diagnostic_change_metrics:
            summary["diagnoses_added"] = self.diagnostic_change_metrics.get(
                "diagnoses_added", 0
            )
            summary["diagnoses_removed"] = self.diagnostic_change_metrics.get(
                "diagnoses_removed", 0
            )
            summary["diagnoses_maintained"] = self.diagnostic_change_metrics.get(
                "diagnoses_maintained", 0
            )
            summary["exact_accuracy_change"] = self.diagnostic_change_metrics.get(
                "exact_accuracy_change", 0.0
            )
            summary["approx_accuracy_change"] = self.diagnostic_change_metrics.get(
                "approx_accuracy_change", 0.0
            )
            summary["confidence_score_change"] = self.diagnostic_change_metrics.get(
                "confidence_score_change", 0.0
            )
            summary["confidence_shifts"] = self.diagnostic_change_metrics.get(
                "confidence_shifts", 0.0
            )
            summary["total_change_magnitude"] = self.diagnostic_change_metrics.get(
                "total_change_magnitude", 0.0
            )

        # Key counts
        summary["matched_keys_count"] = len(self.matched_keys)
        summary["unmatched_case_keys_count"] = len(self.unmatched_case_keys)
        summary["unmatched_request_keys_count"] = len(self.unmatched_request_keys)

        summary.update(self.confidence_values)
        return summary
