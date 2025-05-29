import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

import spacy
import txtai
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from vivabench.ontology.schema import (
    ClinicalCase,
    InvestigationResult,
    PhysicalExamination,
    Symptom,
)
from vivabench.ontology.synonyms import ALL_IMG_SYNONYMS, ALL_IX_SYNONYMS
from vivabench.prompts.mapper import (
    HX_MAP_SYSTEM,
    HX_RETREIVAL_TEMPLATE,
    IMAGING_RETRIEVAL_SYSTEM,
    IMAGING_TEMPLATE,
    LAB_RETRIEVAL_SYSTEM,
    LAB_TEMPLATE,
    PHYS_RETRIEVAL_SYSTEM,
    PHYS_RETRIEVAL_TEMPLATE,
)
from vivabench.utils import prettify, remove_json_markdown, smart_capitalize

class ActionMapper(ABC):
    """A mapper maps any free-text query into a set of keys. This can be done via an LLM or determinsitically
    with traditional NLP methods. Overall, history and physical examinations are harder to parse with pre-defined
    keys, but investigations / imaging can be mostly with cosine similarity on entities
    """

    @abstractmethod
    def __init__(self, clincase: ClinicalCase):
        pass

    @abstractmethod
    def map_history_requests(self, query: str):
        pass

    @abstractmethod
    def map_physical_requests(self, query: str):
        pass

    @abstractmethod
    def map_investigation_requests(self, query: str):
        pass

    @abstractmethod
    def map_imaging_requests(self, query: str):
        pass


class DeterminsticMapper(ActionMapper):

    def __init__(
        self,
        clincase: ClinicalCase = None,
        snomed_embeddings_path="./medical/snomed_embeddings",
    ):
        self.snomed_embeddings = txtai.Embeddings(
            path="neuml/pubmedbert-base-embeddings", content=True
        )

        self.snomed_embeddings.load(snomed_embeddings_path)

        self.sx_mapping = txtai.Embeddings(
            path="neuml/pubmedbert-base-embeddings", content=True
        )

        self.phys_mapping = txtai.Embeddings(
            path="neuml/pubmedbert-base-embeddings", content=True
        )

        self.sx_keys = []
        self.phys_keys = []

        self.nlp = spacy.load("en_core_sci_md")

        self.ix_keyword_mapping = defaultdict(set)
        for k, v in ALL_IX_SYNONYMS.items():
            for _v in v:
                self.ix_keyword_mapping[_v].add(k)

        self.img_keyword_mapping = defaultdict(set)
        for k, v in ALL_IMG_SYNONYMS.items():
            for _v in v:
                self.img_keyword_mapping[_v].add(k)

        if clincase:
            self.load_case(clincase)

    def load_case(self, clincase: ClinicalCase):

        self.clincase = clincase

        if self.sx_keys:
            self.sx_mapping.delete(range(len(self.sx_keys)))
        if self.phys_keys:
            self.phys_mapping.delete(range(len(self.phys_keys)))

        self.sx_keys = list(clincase.history.dict().keys())
        self.sx_vals = list(v.lower() for v in clincase.history.dict().values())

        self.phys_keys = list(clincase.physical.dict().keys())
        self.phys_vals = list(
            v.split(":")[0].lower() for v in clincase.physical.dict().values()
        )

        self.sx_mapping.index(self.sx_vals)
        self.phys_mapping.index(self.phys_vals)

    def map_history_requests(self, query):

        SNOMED_THRESHOLD = 0.8
        MATCH_THRESHOLD = 0.6

        mapped_requests = {"matched": [], "unmatched": []}

        for freetext_term in self.nlp(str(query)).ents:
            mapped_terms = set()

            freetext_term = str(freetext_term).lower()

            search_results = self.sx_mapping.search(str(freetext_term), limit=5)
            candidate_terms = [
                int(t["id"]) for t in search_results if t["score"] > MATCH_THRESHOLD
            ]

            mapped_terms.update(set(candidate_terms))

            # Search through SNOMED as well
            search_results = self.snomed_embeddings.search(str(freetext_term), limit=5)
            candidate_terms = [
                t["text"] for t in search_results if t["score"] > SNOMED_THRESHOLD
            ]

            for c in candidate_terms:
                search_results = self.sx_mapping.search(c, limit=5)

                candidate_terms = [
                    int(t["id"]) for t in search_results if t["score"] > MATCH_THRESHOLD
                ]
                mapped_terms.update(set(candidate_terms))

            if mapped_terms:
                for mapped_idx in mapped_terms:
                    mapped_requests["matched"].append(
                        {"query": freetext_term, "key": self.sx_keys[mapped_idx]}
                    )
            else:
                mapped_requests["unmatched"].append(
                    {
                        "query": freetext_term,
                        "key": freetext_term.lower().replace(" ", "_"),
                    }
                )

        return mapped_requests

    def map_physical_requests(self, query):

        SNOMED_THRESHOLD = 0.8
        MATCH_THRESHOLD = 0.6

        mapped_requests = {"matched": [], "unmatched": []}

        for freetext_term in self.nlp(str(query)).ents:
            mapped_terms = set()

            freetext_term = str(freetext_term).lower()

            search_results = self.phys_mapping.search(str(freetext_term), limit=5)
            candidate_terms = [
                int(t["id"]) for t in search_results if t["score"] > MATCH_THRESHOLD
            ]

            mapped_terms.update(set(candidate_terms))

            # Search through SNOMED as well
            search_results = self.snomed_embeddings.search(str(freetext_term), limit=5)
            candidate_terms = [
                t["text"] for t in search_results if t["score"] > SNOMED_THRESHOLD
            ]

            for c in candidate_terms:
                search_results = self.phys_mapping.search(c, limit=5)

                candidate_terms = [
                    int(t["id"]) for t in search_results if t["score"] > MATCH_THRESHOLD
                ]
                mapped_terms.update(set(candidate_terms))

            if mapped_terms:
                for mapped_idx in mapped_terms:
                    mapped_requests["matched"].append(
                        {"query": freetext_term, "key": self.phys_keys[mapped_idx]}
                    )
            else:
                mapped_requests["unmatched"].append(
                    {
                        "query": freetext_term,
                        "key": freetext_term.lower().replace(" ", "_"),
                    }
                )

        return mapped_requests

    def map_investigation_requests(self, query):

        mapped_requests = {"matched": [], "unmatched": []}
        for freetext_term in self.nlp(str(query)).ents:
            freetext_term = str(freetext_term)
            mapped_terms = self.ix_keyword_mapping.get(freetext_term, set())

            mapped_keys = mapped_terms.intersection(
                set(self.clincase.investigations.keys())
            )

            if mapped_keys:

                for mapped_key in mapped_keys:
                    mapped_requests["matched"].append(
                        {"query": freetext_term, "key": mapped_key}
                    )

            else:
                mapped_requests["unmatched"].append(
                    {"query": freetext_term, "key": freetext_term}
                )

        return mapped_requests

    def map_imaging_requests(self, query):

        mapped_requests = {"matched": [], "unmatched": []}
        for freetext_term in self.nlp(str(query)).ents:
            freetext_term = str(freetext_term)
            mapped_terms = self.img_keyword_mapping.get(freetext_term, set())

            mapped_keys = mapped_terms.intersection(set(self.clincase.imaging.keys()))

            if mapped_keys:

                for mapped_key in mapped_keys:
                    mapped_requests["matched"].append(
                        {"query": freetext_term, "key": mapped_key}
                    )

            else:
                mapped_requests["unmatched"].append(
                    {"query": freetext_term, "key": freetext_term}
                )

        return mapped_requests


class LLMMapper(ActionMapper):

    def __init__(self, clincase: ClinicalCase, model: BaseChatModel):
        self.model = model
        self.clincase = clincase

        self.history_asked = False
        self.physical_performed = False

        self.token_usage = 0

    def get_keys(self, query):
        response = self.model.invoke(query)
        self.token_usage += response.usage_metadata["total_tokens"]

        response_stripped = remove_json_markdown(response.content)

        try:
            response_parsed = json.loads(response_stripped)
        except Exception as e:
            raise ValueError(e, response_stripped)

        return response_parsed

    def map_history_requests(self, query):

        query = [
            HX_MAP_SYSTEM,
            HumanMessage(
                HX_RETREIVAL_TEMPLATE.format(
                    query=query,
                    keys=str(self.clincase.history.keys()),
                    chief_complaint=self.clincase.history.chief_complaint,
                )
            ),
        ]

        return self.get_keys(query)

    def map_physical_requests(self, query):

        query = [
            PHYS_RETRIEVAL_SYSTEM,
            HumanMessage(
                PHYS_RETRIEVAL_TEMPLATE.format(
                    query=query, keys=str(self.clincase.physical.keys())
                )
            ),
        ]

        return self.get_keys(query)

    def map_investigation_requests(self, query):

        query = [
            LAB_RETRIEVAL_SYSTEM,
            HumanMessage(
                LAB_TEMPLATE.format(
                    query=query, items=str(self.clincase.investigations.dict())
                )
            ),
        ]

        return self.get_keys(query)

    def map_imaging_requests(self, query):

        query = [
            IMAGING_RETRIEVAL_SYSTEM,
            HumanMessage(
                IMAGING_TEMPLATE.format(
                    query=query, keys=str(self.clincase.imaging_keys())
                )
            ),
        ]

        return self.get_keys(query)
