"""NLP pipeline to filter for potentially relevant cases from PubMed first, before further human review
Parallelized version with batch processing"""

import gc
import multiprocessing as mp
import os
import re
from functools import partial

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Configure multiprocessing
NUM_PROCESSES = max(1, mp.cpu_count() - 1)  # Leave one CPU free
BATCH_SIZE = 5000  # Process this many rows at once


def initial_filter(reports_df):
    """
    First-pass filtering based on keywords and patterns
    """
    # Diagnostic focus keywords
    diagnostic_keywords = [
        "diagnosis",
        "diagnostic",
        "differential diagnosis",
        "clinical presentation",
        "presenting with",
        "presented with",
        "case of",
        "rare case",
        "unusual presentation",
        "diagnostic challenge",
        "diagnostic dilemma",
        "diagnostic workup",
        "clinical findings",
    ]

    # History/physical examination keywords
    history_pe_keywords = [
        "medical history",
        "past medical history",
        "family history",
        "social history",
        "physical examination",
        "vital signs",
        "on examination",
        "clinical examination",
        "review of systems",
        "chief complaint",
        "presenting complaint",
        "symptoms",
        "signs",
        "physical findings",
    ]

    # Imaging/labs keywords
    imaging_lab_keywords = [
        "laboratory",
        "imaging",
        "radiograph",
        "x-ray",
        "CT",
        "MRI",
        "ultrasound",
        "blood test",
        "serum",
        "biopsy",
        "pathology",
        "histopathology",
        "biochemistry",
        "hematology",
        "complete blood count",
        "CBC",
        "electrolytes",
        "glucose",
    ]

    # Create regex patterns
    diagnostic_pattern = "|".join(diagnostic_keywords)
    history_pe_pattern = "|".join(history_pe_keywords)
    imaging_lab_pattern = "|".join(imaging_lab_keywords)

    # Apply filters
    reports_df["has_diagnostic"] = reports_df["patient"].str.contains(
        diagnostic_pattern, case=False, regex=True
    )
    reports_df["has_history_pe"] = reports_df["patient"].str.contains(
        history_pe_pattern, case=False, regex=True
    )
    reports_df["has_imaging_lab"] = reports_df["patient"].str.contains(
        imaging_lab_pattern, case=False, regex=True
    )

    # Calculate a simple score
    reports_df["filter_score"] = (
        reports_df["has_diagnostic"].astype(int) * 3
        + reports_df["has_history_pe"].astype(int) * 2
        + reports_df["has_imaging_lab"].astype(int)
    )

    # Filter reports that meet minimum criteria (has diagnostic focus and at least history/PE or imaging/labs)
    filtered_reports = reports_df[
        (reports_df["has_diagnostic"])
        & (reports_df["has_history_pe"] | reports_df["has_imaging_lab"])
    ]

    # Sort by score
    filtered_reports = filtered_reports.sort_values("filter_score", ascending=False)

    return filtered_reports


def advanced_filter(filtered_reports):
    """
    Second-pass filtering using more sophisticated NLP techniques
    """
    # Check for section headers that indicate detailed patient information
    section_headers = [
        r"case (?:presentation|report)",
        r"patient (?:presentation|history)",
        r"clinical (?:presentation|history|findings)",
        r"physical examination",
        r"laboratory (?:findings|results|investigations)",
        r"imaging (?:findings|results|studies)",
        r"diagnostic (?:workup|evaluation|assessment)",
    ]

    header_pattern = "|".join(section_headers)

    # Count the number of section headers
    filtered_reports["section_count"] = filtered_reports["patient"].apply(
        lambda x: len(re.findall(header_pattern, x, re.IGNORECASE))
    )

    # Check for structured data patterns (like lab values with units)
    lab_value_pattern = (
        r"\b\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L|g/dL|U/L|ng/mL|μg/L|mmHg|bpm|°C|cm|mm)\b"
    )
    filtered_reports["lab_value_count"] = filtered_reports["patient"].apply(
        lambda x: len(re.findall(lab_value_pattern, x))
    )

    # Check for temporal expressions (indicating detailed history)
    temporal_pattern = r"\b(?:for|over|during|after|before|since|past|previous|last)\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b"
    filtered_reports["temporal_expr_count"] = filtered_reports["patient"].apply(
        lambda x: len(re.findall(temporal_pattern, x, re.IGNORECASE))
    )

    # Update score with these new metrics
    filtered_reports["advanced_score"] = (
        filtered_reports["filter_score"]
        + filtered_reports["section_count"] * 2
        + filtered_reports["lab_value_count"] * 0.5
        + filtered_reports["temporal_expr_count"]
    )

    # Sort by the advanced score
    filtered_reports = filtered_reports.sort_values("advanced_score", ascending=False)

    return filtered_reports


def content_density_analysis(filtered_reports):
    """
    Analyze the density of relevant clinical information
    """
    # Calculate text length (longer texts might have more details)
    filtered_reports["text_length"] = filtered_reports["patient"].str.len()

    # Calculate information density using TF-IDF for medical terms
    medical_terms = [
        # Diagnostic terms
        "diagnosis",
        "differential",
        "etiology",
        "pathology",
        "syndrome",
        # Symptom terms
        "pain",
        "fever",
        "fatigue",
        "nausea",
        "vomiting",
        "diarrhea",
        "cough",
        "dyspnea",
        # Physical exam terms
        "auscultation",
        "palpation",
        "percussion",
        "inspection",
        "reflexes",
        # Vital signs
        "blood pressure",
        "heart rate",
        "respiratory rate",
        "temperature",
        "oxygen saturation",
        # Lab terms
        "hemoglobin",
        "leukocytes",
        "platelets",
        "creatinine",
        "glucose",
        "sodium",
        "potassium",
    ]

    # Create a custom vectorizer that focuses on medical terms
    vectorizer = TfidfVectorizer(
        vocabulary=medical_terms,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )

    # Transform the abstracts
    tfidf_matrix = vectorizer.fit_transform(filtered_reports["patient"])

    # Calculate the sum of TF-IDF scores as a measure of medical content density
    filtered_reports["medical_term_density"] = tfidf_matrix.sum(axis=1).A1

    # Normalize by text length to get true density
    filtered_reports["normalized_density"] = (
        filtered_reports["medical_term_density"]
        / filtered_reports["text_length"]
        * 1000
    )

    # Update final score
    filtered_reports["final_score"] = (
        filtered_reports["advanced_score"] * 0.7
        + filtered_reports["normalized_density"] * 0.3
    )

    # Sort by final score
    filtered_reports = filtered_reports.sort_values("final_score", ascending=False)

    return filtered_reports


def structural_analysis(filtered_reports):
    """
    Analyze the structure of case reports to identify well-organized ones
    """
    # Define patterns for well-structured case reports
    intro_pattern = r"\b(?:we|I)\s+(?:present|report|describe)\s+a\s+(?:case|patient)\b"
    conclusion_pattern = r"\b(?:in conclusion|to conclude|we conclude|this case demonstrates|this case highlights)\b"

    # Check for presence of introduction and conclusion
    filtered_reports["has_intro"] = filtered_reports["patient"].str.contains(
        intro_pattern, case=False, regex=True
    )
    filtered_reports["has_conclusion"] = filtered_reports["patient"].str.contains(
        conclusion_pattern, case=False, regex=True
    )

    # Check for paragraph structure (approximated by newlines or sentence patterns)
    filtered_reports["paragraph_count"] = filtered_reports["patient"].apply(
        lambda x: (
            x.count("\n") + 1
            if "\n" in x
            else max(1, len(re.findall(r"(?<=[.!?])\s+(?=[A-Z])", x)))
        )
    )

    # Check for presence of demographic information
    demographic_pattern = r"\b(?:year[\s-]old|yo|aged|age)\b.{1,20}\b(?:man|woman|male|female|boy|girl|patient)\b"
    filtered_reports["has_demographics"] = filtered_reports["patient"].str.contains(
        demographic_pattern, case=False, regex=True
    )

    # Update structure score
    filtered_reports["structure_score"] = (
        filtered_reports["has_intro"].astype(int) * 2
        + filtered_reports["has_conclusion"].astype(int) * 1
        + filtered_reports["has_demographics"].astype(int) * 2
        + filtered_reports["paragraph_count"].clip(1, 5)
        / 5
        * 3  # Normalize to max of 3 points
    )

    # Update final score with structure consideration
    filtered_reports["final_score"] = (
        filtered_reports["final_score"] * 0.8
        + filtered_reports["structure_score"] * 0.2
    )

    return filtered_reports


def missed_diagnosis_filter(reports_df):
    """
    Filter for cases involving diagnostic errors, delays, or challenges
    """
    # Keywords related to diagnostic errors or challenges
    diagnostic_error_keywords = [
        "misdiagnosis",
        "missed diagnosis",
        "delayed diagnosis",
        "diagnostic error",
        "diagnostic challenge",
        "diagnostic dilemma",
        "diagnostic pitfall",
        "initially diagnosed as",
        "initially misdiagnosed",
        "incorrect diagnosis",
        "failure to diagnose",
        "overlooked",
        "diagnostic uncertainty",
        "masquerading as",
        "mimicking",
        "mistaken for",
        "misinterpreted as",
        "diagnostic delay",
        "diagnostic failure",
        "diagnostic mistake",
    ]

    # Create pattern
    error_pattern = "|".join(diagnostic_error_keywords)

    # Apply filter
    reports_df["has_diagnostic_error"] = reports_df["patient"].str.contains(
        error_pattern, case=False, regex=True
    )

    # Boost score for cases with diagnostic errors
    reports_df["diagnostic_error_score"] = (
        reports_df["has_diagnostic_error"].astype(int) * 5
    )

    return reports_df


def atypical_presentation_filter(reports_df):
    """
    Filter for common conditions with atypical presentations
    """
    # Common conditions
    common_conditions = [
        "pneumonia",
        "myocardial infarction",
        "appendicitis",
        "diabetes",
        "hypertension",
        "stroke",
        "pulmonary embolism",
        "deep vein thrombosis",
        "asthma",
        "COPD",
        "urinary tract infection",
        "cellulitis",
        "meningitis",
        "sepsis",
        "heart failure",
        "pancreatitis",
        "cholecystitis",
        "diverticulitis",
        "pyelonephritis",
    ]

    # Atypical presentation modifiers
    atypical_modifiers = [
        "atypical",
        "unusual",
        "rare",
        "uncommon",
        "non-classic",
        "non-typical",
        "unexpected",
        "misleading",
        "deceptive",
        "subtle",
        "silent",
        "occult",
        "without typical",
        "without classic",
        "without characteristic",
        "atypically presenting",
        "unusual presentation of",
        "rare presentation of",
    ]

    # Create patterns
    condition_pattern = "|".join(common_conditions)
    atypical_pattern = "|".join(atypical_modifiers)

    # Check for common conditions
    reports_df["has_common_condition"] = reports_df["patient"].str.contains(
        condition_pattern, case=False, regex=True
    )

    # Check for atypical modifiers
    reports_df["has_atypical_modifier"] = reports_df["patient"].str.contains(
        atypical_pattern, case=False, regex=True
    )

    # Check for both in proximity (within 10 words)
    def check_proximity(text):
        text_lower = text.lower()
        for condition in common_conditions:
            if condition in text_lower:
                condition_pos = text_lower.find(condition)
                window = text_lower[
                    max(0, condition_pos - 50) : min(
                        len(text_lower), condition_pos + 50
                    )
                ]
                for modifier in atypical_modifiers:
                    if modifier in window:
                        return True
        return False

    # Apply proximity check (this is more computationally intensive, so only apply to rows that have both)
    potential_atypical = reports_df[
        reports_df["has_common_condition"] & reports_df["has_atypical_modifier"]
    ]
    if not potential_atypical.empty:
        potential_atypical["atypical_proximity"] = potential_atypical["patient"].apply(
            check_proximity
        )

        # Update the main dataframe
        reports_df.loc[potential_atypical.index, "atypical_proximity"] = (
            potential_atypical["atypical_proximity"]
        )
        reports_df["atypical_proximity"] = reports_df["atypical_proximity"].fillna(
            False
        )
    else:
        reports_df["atypical_proximity"] = False

    # Score for atypical presentations
    reports_df["atypical_score"] = (
        reports_df["has_common_condition"].astype(int) * 1
        + reports_df["has_atypical_modifier"].astype(int) * 2
        + reports_df["atypical_proximity"].astype(int) * 4
    )

    return reports_df


def serious_condition_filter(reports_df):
    """
    Filter for potentially missed serious conditions like PE
    """
    # List of serious conditions that are commonly missed
    serious_conditions = [
        "pulmonary embolism",
        "PE",
        "aortic dissection",
        "subarachnoid hemorrhage",
        "SAH",
        "meningitis",
        "endocarditis",
        "myocardial infarction",
        "MI",
        "STEMI",
        "NSTEMI",
        "stroke",
        "CVA",
        "ectopic pregnancy",
        "appendicitis",
        "sepsis",
        "necrotizing fasciitis",
        "cauda equina",
        "testicular torsion",
        "abdominal aortic aneurysm",
        "AAA",
        "epidural hematoma",
        "subdural hematoma",
        "tension pneumothorax",
    ]

    # Phrases indicating these conditions might be missed
    missed_indicators = [
        "missed",
        "delayed",
        "overlooked",
        "not initially diagnosed",
        "not recognized",
        "failure to diagnose",
        "failure to recognize",
        "undiagnosed",
        "unrecognized",
        "initially treated as",
        "initially diagnosed as",
        "misdiagnosed as",
    ]

    # Create patterns
    serious_pattern = "|".join(serious_conditions)
    missed_pattern = "|".join(missed_indicators)

    # Apply filters
    reports_df["has_serious_condition"] = reports_df["patient"].str.contains(
        serious_pattern, case=False, regex=True
    )
    reports_df["has_missed_indicator"] = reports_df["patient"].str.contains(
        missed_pattern, case=False, regex=True
    )

    # Check for both in the same document
    reports_df["potential_missed_serious"] = (
        reports_df["has_serious_condition"] & reports_df["has_missed_indicator"]
    )

    # Score for missed serious conditions
    reports_df["serious_condition_score"] = (
        reports_df["has_serious_condition"].astype(int) * 2
        + reports_df["has_missed_indicator"].astype(int) * 1
        + reports_df["potential_missed_serious"].astype(int) * 5
    )

    return reports_df


def semi_common_presentation_filter(reports_df):
    """
    Filter for semi-common presentations that are still clinically relevant
    """
    # Semi-common presentations or conditions
    semi_common_conditions = [
        "pericarditis",
        "endocarditis",
        "myocarditis",
        "vasculitis",
        "sarcoidosis",
        "polymyalgia rheumatica",
        "temporal arteritis",
        "giant cell arteritis",
        "Guillain-Barré syndrome",
        "multiple sclerosis",
        "transverse myelitis",
        "thyroiditis",
        "adrenal insufficiency",
        "Cushing syndrome",
        "acromegaly",
        "hemochromatosis",
        "Wilson disease",
        "celiac disease",
        "inflammatory bowel disease",
        "autoimmune hepatitis",
        "primary biliary cholangitis",
        "primary sclerosing cholangitis",
        "interstitial lung disease",
        "sarcoidosis",
        "pulmonary hypertension",
        "pheochromocytoma",
        "carcinoid syndrome",
        "amyloidosis",
    ]

    # Create pattern
    semi_common_pattern = "|".join(semi_common_conditions)

    # Apply filter
    reports_df["has_semi_common"] = reports_df["patient"].str.contains(
        semi_common_pattern, case=False, regex=True
    )

    # Score for semi-common conditions
    reports_df["semi_common_score"] = reports_df["has_semi_common"].astype(int) * 3

    return reports_df


def first_presentation_filter(reports_df):
    """
    Filter to prioritize first presentations and exclude transfers/follow-ups
    """
    # Negative patterns indicating transfers or follow-ups
    transfer_patterns = [
        r"transferred to",
        r"transferred from",
        r"was transferred",
        r"referred to",
        r"was referred",
        r"referred from",
        r"follow-up",
        r"follow up",
        r"followup",
        r"readmission",
        r"re-admission",
        r"readmitted",
        r"previous admission",
        r"prior admission",
        r"after emergency",
        r"post-operative",
        r"postoperative",
        r"after surgery",
        r"following surgery",
        r"previously diagnosed",
        r"previously treated",
        r"known case of",
        r"known history of",
        r"recurrent",
        r"relapse",
        r"relapsing",
    ]

    # Positive patterns indicating first presentations
    first_presentation_patterns = [
        r"first presentation",
        r"initial presentation",
        r"presenting complaint",
        r"presented to (the)? emergency",
        r"presented to (the)? ED",
        r"presented to (the)? hospital",
        r"presented to (the)? clinic",
        r"presented with",
        r"admission",
        r"chief complaint",
        r"came to (the)? emergency",
        r"came to (the)? ED",
        r"came to (the)? hospital",
        r"came to (the)? clinic",
        r"arrived at (the)? emergency",
        r"arrived at (the)? ED",
        r"first episode",
        r"first occurrence",
        r"first manifestation",
        r"new onset",
        r"newly diagnosed",
        r"initial diagnosis",
        r"first visit",
        r"initial visit",
        r"first consultation",
    ]

    # Create combined negative pattern
    transfer_pattern = "|".join(transfer_patterns)

    # Create combined positive pattern
    first_pattern = "|".join(first_presentation_patterns)

    # Apply filters
    reports_df["has_transfer_indicator"] = reports_df["patient"].str.contains(
        transfer_pattern, case=False, regex=True
    )
    reports_df["has_first_presentation"] = reports_df["patient"].str.contains(
        first_pattern, case=False, regex=True
    )

    # Calculate presentation score
    # High positive score for first presentations, negative penalty for transfers
    reports_df["presentation_score"] = (
        reports_df["has_first_presentation"].astype(int) * 4
        - reports_df["has_transfer_indicator"].astype(int) * 3
    )

    return reports_df


def enhanced_presentation_context(reports_df):
    """
    More nuanced analysis of the presentation context
    """

    # Function to analyze the first few sentences for presentation context
    def analyze_intro(text):
        # Get first 3 sentences or first 300 characters, whichever is longer
        sentences = re.split(r"[.!?]", text)
        intro = " ".join(sentences[: min(3, len(sentences))])
        if len(intro) < 300:
            intro = text[: min(300, len(text))]

        # Check for transfer indicators in the intro
        transfer_words = [
            "transferred",
            "referral",
            "referred",
            "previous",
            "follow-up",
            "readmission",
        ]
        transfer_in_intro = any(word in intro.lower() for word in transfer_words)

        # Check for first presentation indicators in the intro
        first_words = [
            "presented",
            "presentation",
            "admitted",
            "admission",
            "came to",
            "arrived",
        ]
        first_in_intro = any(word in intro.lower() for word in first_words)

        # Check for temporal indicators suggesting first episode
        temporal_first = re.search(
            r"first time|initial episode|first episode|first onset|sudden onset",
            intro,
            re.IGNORECASE,
        )

        # Score based on these factors
        score = 0
        if transfer_in_intro:
            score -= 3
        if first_in_intro:
            score += 2
        if temporal_first:
            score += 3

        return score

    # Apply the analysis
    reports_df["intro_context_score"] = reports_df["patient"].apply(analyze_intro)

    # Update presentation score
    reports_df["presentation_score"] = (
        reports_df["presentation_score"] + reports_df["intro_context_score"]
    )

    return reports_df


def exclude_icu_transfers(reports_df):
    """
    Specifically target and exclude ICU transfer cases
    """
    # Patterns indicating ICU transfers
    icu_transfer_patterns = [
        r"transferred to (our|the) (ICU|intensive care)",
        r"transferred to (our|the) (ICU|intensive care)",
        r"admitted to (the|our) (ICU|intensive care) (after|following)",
        r"(ICU|intensive care) (transfer|admission) (after|following)",
        r"(after|following) emergency .{1,30} (transferred|admitted) to (ICU|intensive care)",
    ]

    # Create combined pattern
    icu_pattern = "|".join(icu_transfer_patterns)

    # Apply filter
    reports_df["is_icu_transfer"] = reports_df["patient"].str.contains(
        icu_pattern, case=False, regex=True
    )

    # Apply strong penalty for ICU transfer cases
    reports_df["presentation_score"] = reports_df["presentation_score"] - (
        reports_df["is_icu_transfer"].astype(int) * 5
    )

    return reports_df


def exclude_post_procedure_cases(reports_df):
    """
    Filter out cases that are primarily about post-procedure complications
    """
    # Patterns indicating post-procedure cases
    post_procedure_patterns = [
        r"(after|following|post) (surgery|procedure|operation|intervention)",
        r"(after|following|post)(operative|procedural|surgical)",
        r"complication (of|following|after)",
        r"(surgery|procedure|operation) complication",
        r"(iatrogenic|procedure-related|surgery-related)",
        r"(days|weeks) (after|following|post) (surgery|procedure|operation)",
    ]

    # Create combined pattern
    post_procedure_pattern = "|".join(post_procedure_patterns)

    # Apply filter
    reports_df["is_post_procedure"] = reports_df["patient"].str.contains(
        post_procedure_pattern, case=False, regex=True
    )

    # Apply penalty for post-procedure cases
    reports_df["presentation_score"] = reports_df["presentation_score"] - (
        reports_df["is_post_procedure"].astype(int) * 4
    )

    return reports_df


def prioritize_emergency_presentations(reports_df):
    """
    Give higher scores to emergency or urgent presentations
    """
    # Patterns indicating emergency presentations
    emergency_patterns = [
        r"emergency (department|room|ward|admission)",
        r"ED presentation",
        r"ER presentation",
        r"urgent (care|admission)",
        r"acute (presentation|admission)",
        r"presented (acutely|urgently|emergently)",
        r"(rushed|brought) to (the|our) (emergency|ED|ER)",
        r"ambulance",
        r"paramedics",
    ]

    # Create combined pattern
    emergency_pattern = "|".join(emergency_patterns)

    # Apply filter
    reports_df["is_emergency"] = reports_df["patient"].str.contains(
        emergency_pattern, case=False, regex=True
    )

    # Boost score for emergency presentations
    reports_df["presentation_score"] = reports_df["presentation_score"] + (
        reports_df["is_emergency"].astype(int) * 2
    )

    return reports_df


def section_based_analysis(reports_df):
    """
    Analyze case reports for distinct clinical sections
    """
    # Common section headers in case reports
    section_patterns = {
        "history": [
            r"(medical|past medical|clinical) history",
            r"history of (present|current) illness",
            r"presenting complaint",
            r"chief complaint",
            r"history of presentation",
            r"history and examination",
        ],
        "physical_exam": [
            r"physical examination",
            r"clinical examination",
            r"on examination",
            r"physical findings",
            r"vital signs",
            r"examination revealed",
        ],
        "investigations": [
            r"laboratory (findings|results|tests|values|investigations)",
            r"lab (findings|results|tests|values|investigations)",
            r"diagnostic (studies|tests|investigations)",
            r"blood (tests|work|results)",
            r"imaging (studies|results|findings)",
            r"radiologic (studies|findings)",
            r"further testing",
            r"additional (tests|testing|laboratory|investigations)",
        ],
        "diagnosis": [
            r"diagnosis",
            r"diagnostic (assessment|impression)",
            r"clinical diagnosis",
            r"final diagnosis",
            r"differential diagnosis",
        ],
        "treatment": [
            r"treatment",
            r"management",
            r"therapeutic (approach|intervention)",
            r"therapy",
            r"intervention",
        ],
        "outcome": [
            r"outcome",
            r"follow-up",
            r"clinical course",
            r"hospital course",
            r"patient course",
            r"resolution",
            r"recovery",
        ],
    }

    # Create combined patterns for each section
    section_regex = {
        section: "|".join(patterns) for section, patterns in section_patterns.items()
    }

    # Function to detect sections and analyze their content
    def analyze_sections(text):
        results = {}

        # Check for presence of each section
        for section, pattern in section_regex.items():
            results[f"has_{section}_section"] = bool(
                re.search(pattern, text, re.IGNORECASE)
            )

        # Count total number of identifiable sections
        results["section_count"] = sum(
            1 for key, value in results.items() if value and key.startswith("has_")
        )

        # Analyze content between sections (simplified approach)
        # This looks for laboratory values, which are common in investigation sections
        lab_value_pattern = r"\b\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L|g/dL|U/L|IU/L|ng/mL|μg/L|mmHg|bpm|°C|cm|mm)\b"
        results["lab_value_count"] = len(re.findall(lab_value_pattern, text))

        # Look for imaging mentions
        imaging_pattern = r"\b(?:ultrasound|CT|MRI|x-ray|radiograph|imaging|scan)\b"
        results["imaging_mention_count"] = len(
            re.findall(imaging_pattern, text, re.IGNORECASE)
        )

        # Look for physical exam findings
        exam_finding_pattern = r"\b(?:revealed|showed|demonstrated|noted|observed|found|examination)\b.{1,30}\b(?:normal|abnormal|elevated|reduced|increased|decreased|positive|negative)\b"
        results["exam_finding_count"] = len(
            re.findall(exam_finding_pattern, text, re.IGNORECASE)
        )

        # Calculate a section richness score
        results["section_richness"] = (
            results["section_count"] * 2
            + min(results["lab_value_count"], 20)
            * 0.2  # Cap at 20 to avoid overweighting
            + min(results["imaging_mention_count"], 10) * 0.3  # Cap at 10
            + min(results["exam_finding_count"], 15) * 0.2  # Cap at 15
        )

        return results

    # Apply the analysis
    section_results = reports_df["patient"].apply(analyze_sections)

    # Convert results to DataFrame columns
    for report_idx, result_dict in enumerate(section_results):
        for key, value in result_dict.items():
            reports_df.loc[reports_df.index[report_idx], key] = value

    return reports_df


def enhanced_section_content_analysis(reports_df):
    """
    Analyze the content within sections, especially focusing on laboratory and diagnostic information
    """

    # Function to analyze laboratory values in text
    def analyze_lab_values(text):
        # Pattern for lab values with units
        lab_pattern = r"(?:(?:of|was|is|at|to|level|value)\s+)?(\d+(?:\.\d+)?)\s*(?:mg/dL|mmol/L|g/dL|U/L|IU/L|ng/mL|μg/L|mmHg|bpm|°C)"

        # Common lab test names
        lab_tests = [
            "hemoglobin",
            "hematocrit",
            "platelets",
            "white blood cell",
            "WBC",
            "neutrophil",
            "lymphocyte",
            "monocyte",
            "eosinophil",
            "basophil",
            "creatinine",
            "BUN",
            "blood urea nitrogen",
            "GFR",
            "glomerular filtration rate",
            "sodium",
            "potassium",
            "chloride",
            "bicarbonate",
            "calcium",
            "phosphorus",
            "magnesium",
            "glucose",
            "HbA1c",
            "hemoglobin A1c",
            "magnesium",
            "glucose",
            "HbA1c",
            "hemoglobin A1c",
            "albumin",
            "protein",
            "bilirubin",
            "AST",
            "ALT",
            "alkaline phosphatase",
            "ALP",
            "GGT",
            "LDH",
            "lactate dehydrogenase",
            "amylase",
            "lipase",
            "troponin",
            "CK",
            "creatine kinase",
            "CK-MB",
            "BNP",
            "NT-proBNP",
            "ESR",
            "CRP",
            "prothrombin time",
            "PT",
            "INR",
            "PTT",
            "APTT",
            "D-dimer",
            "TSH",
            "T3",
            "T4",
            "free T4",
            "ferritin",
            "iron",
            "TIBC",
        ]

        # Pattern for lab test names with values
        lab_test_pattern = "|".join(lab_tests)
        lab_test_value_pattern = f"({lab_test_pattern}).{{1,30}}?{lab_pattern}"

        # Count lab test mentions with values
        lab_test_count = len(re.findall(lab_test_value_pattern, text, re.IGNORECASE))

        # Count total lab values (with units)
        total_lab_values = len(re.findall(lab_pattern, text))

        return {
            "specific_lab_test_count": lab_test_count,
            "total_lab_value_count": total_lab_values,
        }

    # Function to analyze imaging findings
    def analyze_imaging(text):
        # Imaging modalities
        imaging_modalities = [
            "x-ray",
            "radiograph",
            "CT",
            "computed tomography",
            "MRI",
            "magnetic resonance",
            "ultrasound",
            "sonography",
            "echocardiogram",
            "angiography",
            "PET",
            "nuclear scan",
            "SPECT",
            "fluoroscopy",
        ]

        # Pattern for imaging modalities
        modality_pattern = "|".join(imaging_modalities)

        # Pattern for imaging findings
        finding_pattern = r"(?:{0}).{{1,50}}(?:revealed|showed|demonstrated|noted|found|identified)".format(
            modality_pattern
        )

        # Count imaging findings
        imaging_finding_count = len(re.findall(finding_pattern, text, re.IGNORECASE))

        # Count total imaging mentions
        total_imaging_mentions = len(re.findall(modality_pattern, text, re.IGNORECASE))

        return {
            "imaging_finding_count": imaging_finding_count,
            "total_imaging_mentions": total_imaging_mentions,
        }

    # Apply the analyses
    lab_results = reports_df["patient"].apply(analyze_lab_values)
    imaging_results = reports_df["patient"].apply(analyze_imaging)

    # Convert results to DataFrame columns
    for report_idx, result_dict in enumerate(lab_results):
        for key, value in result_dict.items():
            reports_df.loc[reports_df.index[report_idx], key] = value

    for report_idx, result_dict in enumerate(imaging_results):
        for key, value in result_dict.items():
            reports_df.loc[reports_df.index[report_idx], key] = value

    # Calculate a content richness score
    reports_df["content_richness"] = (
        reports_df["specific_lab_test_count"] * 0.3
        + reports_df["total_lab_value_count"] * 0.2
        + reports_df["imaging_finding_count"] * 0.3
        + reports_df["total_imaging_mentions"] * 0.2
    )

    return reports_df


def paragraph_structure_analysis(reports_df):
    """
    Analyze the paragraph structure of case reports
    """

    def analyze_paragraphs(text):
        # Split into paragraphs (by double newlines or other paragraph separators)
        paragraphs = re.split(r"\n\s*\n|\r\n\s*\r\n", text)
        if len(paragraphs) <= 1:
            # Try splitting by single newlines if no clear paragraphs
            paragraphs = re.split(r"\n|\r\n", text)
            if len(paragraphs) <= 1:
                # As a last resort, try to split by sentences that might start new paragraphs
                paragraphs = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        # Count paragraphs
        paragraph_count = len(paragraphs)

        # Analyze paragraph content
        results = {
            "paragraph_count": paragraph_count,
            "avg_paragraph_length": sum(len(p) for p in paragraphs)
            / max(1, paragraph_count),
            "max_paragraph_length": (
                max(len(p) for p in paragraphs) if paragraphs else 0
            ),
        }

        # Check if paragraphs follow a logical clinical flow
        # (e.g., history -> exam -> investigations -> diagnosis -> treatment)
        clinical_flow_score = 0

        # Look for history in early paragraphs
        if paragraph_count >= 2:
            first_third = " ".join(paragraphs[: max(1, paragraph_count // 3)])
            if re.search(
                r"history|presented|complaint|symptoms", first_third, re.IGNORECASE
            ):
                clinical_flow_score += 2

        # Look for investigations in middle paragraphs
        if paragraph_count >= 3:
            middle_third = " ".join(
                paragraphs[
                    max(1, paragraph_count // 3) : max(2, 2 * paragraph_count // 3)
                ]
            )
            if re.search(
                r"laboratory|test|investigation|finding|imaging",
                middle_third,
                re.IGNORECASE,
            ):
                clinical_flow_score += 2

        # Look for diagnosis/treatment/outcome in later paragraphs
        if paragraph_count >= 3:
            last_third = " ".join(paragraphs[max(2, 2 * paragraph_count // 3) :])
            if re.search(
                r"diagnosis|treatment|management|outcome|follow-up|discharged",
                last_third,
                re.IGNORECASE,
            ):
                clinical_flow_score += 2

        results["clinical_flow_score"] = clinical_flow_score

        return results

    # Apply the analysis
    paragraph_results = reports_df["patient"].apply(analyze_paragraphs)

    # Convert results to DataFrame columns
    for report_idx, result_dict in enumerate(paragraph_results):
        for key, value in result_dict.items():
            reports_df.loc[reports_df.index[report_idx], key] = value

    # Calculate a structure score
    reports_df["paragraph_structure_score"] = (
        reports_df["paragraph_count"].clip(1, 10) * 0.3  # Cap at 10 paragraphs
        + (reports_df["avg_paragraph_length"] / 100).clip(0, 5)
        * 0.2  # Normalize and cap
        + reports_df["clinical_flow_score"] * 0.5  # Clinical flow is most important
    )

    return reports_df


def temporal_sequence_analysis(reports_df):
    """
    Analyze the temporal sequence of events in case reports
    """

    def analyze_temporal_sequence(text):
        # Temporal markers
        initial_markers = [
            "presented",
            "admission",
            "initially",
            "on presentation",
            "at presentation",
            "first",
            "onset",
            "began",
        ]

        subsequent_markers = [
            "further",
            "additional",
            "later",
            "subsequently",
            "follow-up",
            "repeat",
            "next",
            "then",
            "after",
            "following",
        ]

        outcome_markers = [
            "discharged",
            "resolved",
            "improved",
            "recovery",
            "follow-up",
            "remained",
            "continued",
            "persisted",
            "recurred",
        ]

        # Check for presence of markers
        has_initial = any(
            re.search(r"\b{0}\b".format(marker), text, re.IGNORECASE)
            for marker in initial_markers
        )
        has_subsequent = any(
            re.search(r"\b{0}\b".format(marker), text, re.IGNORECASE)
            for marker in subsequent_markers
        )
        has_outcome = any(
            re.search(r"\b{0}\b".format(marker), text, re.IGNORECASE)
            for marker in outcome_markers
        )

        # Check for temporal expressions
        time_expressions = re.findall(
            r"\b(?:for|after|before|during|within|over)\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b",
            text,
            re.IGNORECASE,
        )
        date_expressions = re.findall(
            r"\b(?:on|in)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
            text,
            re.IGNORECASE,
        )

        # Calculate temporal richness
        temporal_richness = (
            has_initial * 2
            + has_subsequent * 2
            + has_outcome * 2
            + min(len(time_expressions), 5) * 0.6  # Cap at 5
            + min(len(date_expressions), 3) * 0.4  # Cap at 3
        )

        return {
            "has_initial_temporal": has_initial,
            "has_subsequent_temporal": has_subsequent,
            "has_outcome_temporal": has_outcome,
            "time_expression_count": len(time_expressions),
            "date_expression_count": len(date_expressions),
            "temporal_richness": temporal_richness,
        }

    # Apply the analysis
    temporal_results = reports_df["patient"].apply(analyze_temporal_sequence)

    # Convert results to DataFrame columns
    for report_idx, result_dict in enumerate(temporal_results):
        for key, value in result_dict.items():
            reports_df.loc[reports_df.index[report_idx], key] = value

    return reports_df


def lab_value_pattern_recognition(reports_df):
    """
    Recognize patterns of laboratory values in text
    """

    def extract_lab_patterns(text):
        # Common lab test patterns with values and units
        lab_patterns = [
            # Liver function tests
            r"(?:AST|aspartate aminotransferase).{1,20}?(\d+)(?:\.\d+)?\s*(?:U/L|IU/L)",
            r"(?:ALT|alanine aminotransferase).{1,20}?(\d+)(?:\.\d+)?\s*(?:U/L|IU/L)",
            r"(?:ALP|alkaline phosphatase).{1,20}?(\d+)(?:\.\d+)?\s*(?:U/L|IU/L)",
            r"(?:bilirubin).{1,20}?(\d+)(?:\.\d+)?\s*(?:mg/dL)",
            r"(?:albumin).{1,20}?(\d+)(?:\.\d+)?\s*(?:g/dL)",
            # Complete blood count
            r"(?:hemoglobin|Hgb|Hb).{1,20}?(\d+)(?:\.\d+)?\s*(?:g/dL)",
            r"(?:hematocrit|Hct).{1,20}?(\d+)(?:\.\d+)?\s*(?:%)",
            r"(?:white blood cell|WBC).{1,20}?(\d+)(?:\.\d+)?\s*(?:K/μL|×10\^9/L)",
            r"(?:platelet|PLT).{1,20}?(\d+)(?:\.\d+)?\s*(?:K/μL|×10\^9/L)",
            # Kidney function
            r"(?:creatinine).{1,20}?(\d+)(?:\.\d+)?\s*(?:mg/dL)",
            r"(?:BUN|blood urea nitrogen).{1,20}?(\d+)(?:\.\d+)?\s*(?:mg/dL)",
            r"(?:GFR|glomerular filtration rate).{1,20}?(\d+)(?:\.\d+)?\s*(?:mL/min)",
            # Electrolytes
            r"(?:sodium|Na).{1,20}?(\d+)(?:\.\d+)?\s*(?:mEq/L|mmol/L)",
            r"(?:potassium|K).{1,20}?(\d+)(?:\.\d+)?\s*(?:mEq/L|mmol/L)",
            r"(?:chloride|Cl).{1,20}?(\d+)(?:\.\d+)?\s*(?:mEq/L|mmol/L)",
            r"(?:bicarbonate|CO2).{1,20}?(\d+)(?:\.\d+)?\s*(?:mEq/L|mmol/L)",
            # Vital signs
            r"(?:temperature).{1,20}?(\d+)(?:\.\d+)?\s*(?:°C|°F)",
            r"(?:heart rate).{1,20}?(\d+)(?:\.\d+)?\s*(?:bpm|beats/min)",
            r"(?:blood pressure).{1,20}?(\d+)/(\d+)\s*(?:mm ?Hg)",
            r"(?:respiratory rate).{1,20}?(\d+)(?:\.\d+)?\s*(?:breaths/min)",
            r"(?:oxygen saturation).{1,20}?(\d+)(?:\.\d+)?\s*(?:%)",
        ]

        # Count matches for each pattern
        lab_counts = {}
        for pattern in lab_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            lab_counts[pattern] = len(matches)

        # Count total lab values found
        total_lab_values = sum(lab_counts.values())

        # Count unique lab test types
        unique_lab_types = sum(1 for count in lab_counts.values() if count > 0)

        return {
            "total_specific_lab_values": total_lab_values,
            "unique_lab_types": unique_lab_types,
            "lab_richness_score": total_lab_values * 0.3 + unique_lab_types * 0.7,
        }

    # Apply the analysis
    lab_pattern_results = reports_df["patient"].apply(extract_lab_patterns)

    # Convert results to DataFrame columns
    for report_idx, result_dict in enumerate(lab_pattern_results):
        for key, value in result_dict.items():
            reports_df.loc[reports_df.index[report_idx], key] = value

    return reports_df


# Parallel processing functions
def process_batch(batch_df, batch_id):
    """Process a single batch of reports"""
    try:
        print(f"Processing batch {batch_id} with {len(batch_df)} reports")

        # Stage 1: Initial keyword-based filtering
        filtered = initial_filter(batch_df)

        # If no reports pass the initial filter, return empty DataFrame
        if len(filtered) == 0:
            print(f"Batch {batch_id}: No reports passed initial filtering")
            return pd.DataFrame()

        # Stage 2: First presentation filtering
        filtered = first_presentation_filter(filtered)
        filtered = enhanced_presentation_context(filtered)
        filtered = exclude_icu_transfers(filtered)
        filtered = exclude_post_procedure_cases(filtered)
        filtered = prioritize_emergency_presentations(filtered)

        # Stage 3: Section-based analysis
        filtered = section_based_analysis(filtered)
        filtered = enhanced_section_content_analysis(filtered)
        filtered = paragraph_structure_analysis(filtered)
        filtered = temporal_sequence_analysis(filtered)
        filtered = lab_value_pattern_recognition(filtered)

        # Stage 4: Clinical relevance filtering
        filtered = missed_diagnosis_filter(filtered)
        filtered = atypical_presentation_filter(filtered)
        filtered = serious_condition_filter(filtered)
        filtered = semi_common_presentation_filter(filtered)

        # Calculate combined clinical relevance score
        filtered["clinical_relevance_score"] = (
            filtered.get("diagnostic_error_score", 0)
            + filtered.get("atypical_score", 0)
            + filtered.get("serious_condition_score", 0)
            + filtered.get("semi_common_score", 0)
        )

        # Stage 5: Advanced NLP filtering
        filtered = advanced_filter(filtered)
        filtered = content_density_analysis(filtered)
        filtered = structural_analysis(filtered)

        # Calculate final combined score with section-based components
        filtered["final_combined_score"] = (
            filtered["final_score"] * 0.2  # Original score components
            + filtered["clinical_relevance_score"] * 0.2  # Clinical relevance
            + filtered["presentation_score"] * 0.1  # Presentation context
            + filtered["section_richness"] * 0.1  # Section structure
            + filtered["content_richness"] * 0.1  # Content within sections
            + filtered["paragraph_structure_score"] * 0.1  # Paragraph structure
            + filtered["temporal_richness"] * 0.1  # Temporal sequence
            + filtered["lab_richness_score"] * 0.1  # Laboratory value patterns
        )

        # Strongly penalize clear transfer cases
        filtered.loc[filtered["is_icu_transfer"], "final_combined_score"] -= 10

        print(
            f"Batch {batch_id}: Completed processing with {len(filtered)} filtered reports"
        )
        return filtered

    except Exception as e:
        print(f"Error processing batch {batch_id}: {str(e)}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()


def filter_case_reports_parallel(reports_df, top_n=5000, batch_size=BATCH_SIZE):
    """
    Enhanced filtering pipeline with parallel processing
    """
    # Download necessary NLTK resources
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

    print(f"Starting with {len(reports_df)} reports")
    print(f"Using {NUM_PROCESSES} processes with batch size {batch_size}")

    # Split the dataframe into batches
    total_rows = len(reports_df)
    num_batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division

    # Create a list to store results
    filtered_results = []

    # Process in smaller chunks to avoid memory issues
    chunk_size = min(50000, total_rows)  # Process at most 50k rows at a time
    num_chunks = (total_rows + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, total_rows)
        chunk_df = reports_df.iloc[chunk_start:chunk_end].copy()

        print(
            f"Processing chunk {chunk_idx+1}/{num_chunks} (rows {chunk_start} to {chunk_end})"
        )

        # Create batches for this chunk
        batches = []
        for i in range(0, len(chunk_df), batch_size):
            batch_end = min(i + batch_size, len(chunk_df))
            batches.append(
                (chunk_df.iloc[i:batch_end].copy(), f"{chunk_idx}-{i//batch_size}")
            )

        # Process batches in parallel
        with mp.Pool(processes=NUM_PROCESSES) as pool:
            batch_results = list(
                tqdm(
                    pool.starmap(process_batch, batches),
                    total=len(batches),
                    desc=f"Processing chunk {chunk_idx+1}",
                )
            )

        # Combine batch results
        for result_df in batch_results:
            if not result_df.empty:
                filtered_results.append(result_df)

        # Clear memory
        del chunk_df, batches, batch_results
        gc.collect()

    # Combine all results
    if not filtered_results:
        print("No reports passed filtering criteria")
        return pd.DataFrame()

    combined_results = pd.concat(filtered_results, ignore_index=True)
    print(f"Combined results: {len(combined_results)} reports")

    # Sort by final combined score and return top N
    combined_results = combined_results.sort_values(
        "final_combined_score", ascending=False
    )
    top_results = combined_results.head(top_n)

    print(f"Returning top {len(top_results)} reports")
    return top_results


def save_checkpoint(df, filename):
    """Save intermediate results to avoid losing progress"""
    df.to_csv(filename, index=False)
    print(f"Saved checkpoint to {filename}")


def load_checkpoint(filename):
    """Load previously saved results"""
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        print(f"Loaded checkpoint from {filename} with {len(df)} rows")
        return df
    return None


def filter_case_reports_with_sections(reports_df, top_n=5000):
    """
    Original function signature maintained for compatibility,
    but now uses the parallel implementation
    """
    return filter_case_reports_parallel(reports_df, top_n)


# Function to process a single text (for use in apply)
def process_single_text(text, function_name):
    """Apply a specific analysis function to a single text"""
    if function_name == "analyze_sections":
        # Common section headers in case reports
        section_patterns = {
            "history": [
                r"(medical|past medical|clinical) history",
                r"history of (present|current) illness",
                r"presenting complaint",
                r"chief complaint",
                r"history of presentation",
                r"history and examination",
            ],
            "physical_exam": [
                r"physical examination",
                r"clinical examination",
                r"on examination",
                r"physical findings",
                r"vital signs",
                r"examination revealed",
            ],
            "investigations": [
                r"laboratory (findings|results|tests|values|investigations)",
                r"lab (findings|results|tests|values|investigations)",
                r"diagnostic (studies|tests|investigations)",
                r"blood (tests|work|results)",
                r"imaging (studies|results|findings)",
                r"radiologic (studies|findings)",
                r"further testing",
                r"additional (tests|testing|laboratory|investigations)",
            ],
            "diagnosis": [
                r"diagnosis",
                r"diagnostic (assessment|impression)",
                r"clinical diagnosis",
                r"final diagnosis",
                r"differential diagnosis",
            ],
            "treatment": [
                r"treatment",
                r"management",
                r"therapeutic (approach|intervention)",
                r"therapy",
                r"intervention",
            ],
            "outcome": [
                r"outcome",
                r"follow-up",
                r"clinical course",
                r"hospital course",
                r"patient course",
                r"resolution",
                r"recovery",
            ],
        }

        # Create combined patterns for each section
        section_regex = {
            section: "|".join(patterns)
            for section, patterns in section_patterns.items()
        }

        results = {}

        # Check for presence of each section
        for section, pattern in section_regex.items():
            results[f"has_{section}_section"] = bool(
                re.search(pattern, text, re.IGNORECASE)
            )

        # Count total number of identifiable sections
        results["section_count"] = sum(
            1 for key, value in results.items() if value and key.startswith("has_")
        )

        # Analyze content between sections (simplified approach)
        lab_value_pattern = r"\b\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L|g/dL|U/L|IU/L|ng/mL|μg/L|mmHg|bpm|°C|cm|mm)\b"
        results["lab_value_count"] = len(re.findall(lab_value_pattern, text))

        # Look for imaging mentions
        imaging_pattern = r"\b(?:ultrasound|CT|MRI|x-ray|radiograph|imaging|scan)\b"
        results["imaging_mention_count"] = len(
            re.findall(imaging_pattern, text, re.IGNORECASE)
        )

        # Look for physical exam findings
        exam_finding_pattern = r"\b(?:revealed|showed|demonstrated|noted|observed|found|examination)\b.{1,30}\b(?:normal|abnormal|elevated|reduced|increased|decreased|positive|negative)\b"
        results["exam_finding_count"] = len(
            re.findall(exam_finding_pattern, text, re.IGNORECASE)
        )

        # Calculate a section richness score
        results["section_richness"] = (
            results["section_count"] * 2
            + min(results["lab_value_count"], 20) * 0.2
            + min(results["imaging_mention_count"], 10) * 0.3
            + min(results["exam_finding_count"], 15) * 0.2
        )

        return results

    elif function_name == "analyze_lab_values":
        # Pattern for lab values with units
        lab_pattern = r"(?:(?:of|was|is|at|to|level|value)\s+)?(\d+(?:\.\d+)?)\s*(?:mg/dL|mmol/L|g/dL|U/L|IU/L|ng/mL|μg/L|mmHg|bpm|°C)"

        # Common lab test names
        lab_tests = [
            "hemoglobin",
            "hematocrit",
            "platelets",
            "white blood cell",
            "WBC",
            "neutrophil",
            "lymphocyte",
            "monocyte",
            "eosinophil",
            "basophil",
            "creatinine",
            "BUN",
            "blood urea nitrogen",
            "GFR",
            "glomerular filtration rate",
            "sodium",
            "potassium",
            "chloride",
            "bicarbonate",
            "calcium",
            "phosphorus",
            "magnesium",
            "glucose",
            "HbA1c",
            "hemoglobin A1c",
            "albumin",
            "protein",
            "bilirubin",
            "AST",
            "ALT",
            "alkaline phosphatase",
            "ALP",
            "GGT",
            "LDH",
            "lactate dehydrogenase",
            "amylase",
            "lipase",
            "troponin",
            "CK",
            "creatine kinase",
            "CK-MB",
            "BNP",
            "NT-proBNP",
            "ESR",
            "CRP",
            "prothrombin time",
            "PT",
            "INR",
            "PTT",
            "APTT",
            "D-dimer",
            "TSH",
            "T3",
            "T4",
            "free T4",
            "ferritin",
            "iron",
            "TIBC",
        ]

        # Pattern for lab test names with values
        lab_test_pattern = "|".join(lab_tests)
        lab_test_value_pattern = f"({lab_test_pattern}).{{1,30}}?{lab_pattern}"

        # Count lab test mentions with values
        lab_test_count = len(re.findall(lab_test_value_pattern, text, re.IGNORECASE))

        # Count total lab values (with units)
        total_lab_values = len(re.findall(lab_pattern, text))

        return {
            "specific_lab_test_count": lab_test_count,
            "total_lab_value_count": total_lab_values,
        }

    # Add more function handlers as needed

    return {}


# Main execution function with better error handling and progress tracking
def main(input_file, output_file, checkpoint_file=None, top_n=5000):
    """Main execution function with checkpointing and error handling"""
    try:
        # Check if checkpoint exists
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"Loading checkpoint from {checkpoint_file}")
            result_df = load_checkpoint(checkpoint_file)
            if result_df is not None and len(result_df) >= top_n:
                print(
                    f"Checkpoint already contains {len(result_df)} results, saving to {output_file}"
                )
                result_df.head(top_n).to_csv(output_file, index=False)
                return

        # Load input data
        print(f"Loading data from {input_file}")
        reports_df = pd.read_csv(input_file)
        print(f"Loaded {len(reports_df)} reports")

        # Process the reports
        result_df = filter_case_reports_parallel(reports_df, top_n=top_n)

        # Save results
        if not result_df.empty:
            print(f"Saving {len(result_df)} filtered reports to {output_file}")
            result_df.to_csv(output_file, index=False)

            # Save checkpoint
            if checkpoint_file:
                save_checkpoint(result_df, checkpoint_file)
        else:
            print("No reports passed filtering criteria")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter case reports from PubMed")
    parser.add_argument("input_file", help="Input CSV file with case reports")
    parser.add_argument("output_file", help="Output CSV file for filtered reports")
    parser.add_argument("--checkpoint", help="Checkpoint file to save/resume progress")
    parser.add_argument(
        "--top_n", type=int, default=5000, help="Number of top reports to keep"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for processing"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=NUM_PROCESSES,
        help="Number of processes to use",
    )

    args = parser.parse_args()

    # Update global settings
    BATCH_SIZE = args.batch_size
    NUM_PROCESSES = args.processes

    main(args.input_file, args.output_file, args.checkpoint, args.top_n)
