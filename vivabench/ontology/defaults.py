import random

DEFAULT_VALUES = {
    "blood:mean_corpuscular_volume": {
        "name": "MCV",
        "unit": "fL",
        "lower": 80.0,
        "upper": 100.0,
    },
    "blood:platelets": {
        "name": "Platelets",
        "unit": "×10^9/L",
        "lower": 150,
        "upper": 400,
    },
    "blood:hemoglobin": {
        "name": "Hb",
        "unit": "g/dL",
        "lower": 13.5,
        "upper": 17.5,
    },
    "blood:white_blood_cell_count": {
        "name": "WBC",
        "unit": "×10^9/L",
        "lower": 4.0,
        "upper": 11.0,
    },
    "blood:chloride": {
        "name": "Cl⁻",
        "unit": "mmol/L",
        "lower": 98,
        "upper": 106,
    },
    "blood:bicarbonate": {
        "name": "HCO₃⁻",
        "unit": "mmol/L",
        "lower": 22,
        "upper": 29,
    },
    "blood:sodium": {
        "name": "Na⁺",
        "unit": "mmol/L",
        "lower": 135,
        "upper": 145,
    },
    "blood:potassium": {
        "name": "K⁺",
        "unit": "mmol/L",
        "lower": 3.5,
        "upper": 5.1,
    },
    "blood:blood_urea_nitrogen": {
        "name": "BUN",
        "unit": "mg/dL",
        "lower": 7,
        "upper": 20,
    },
    "blood:creatinine": {
        "name": "Creatinine",
        "unit": "mg/dL",
        "lower": 0.6,
        "upper": 1.3,
    },
    "blood:glucose": {
        "name": "Glucose",
        "unit": "mg/dL",
        "lower": 70,
        "upper": 99,
    },
    "blood:albumin": {
        "name": "Albumin",
        "unit": "g/dL",
        "lower": 3.5,
        "upper": 5.0,
    },
    "blood:alkaline_phosphatase": {
        "name": "ALP",
        "unit": "U/L",
        "lower": 44,
        "upper": 147,
    },
    "blood:alanine_aminotransferase": {
        "name": "ALT",
        "unit": "U/L",
        "lower": 7,
        "upper": 56,
    },
    "blood:aspartate_aminotransferase": {
        "name": "AST",
        "unit": "U/L",
        "lower": 10,
        "upper": 40,
    },
    "blood:gamma_glutamyl_transferase": {
        "name": "GGT",
        "unit": "U/L",
        "lower": 0,
        "upper": 51,
    },
    "blood:total_bilirubin": {
        "name": "Total Bilirubin",
        "unit": "mg/dL",
        "lower": 0.1,
        "upper": 1.2,
    },
    "blood:direct_bilirubin": {
        "name": "Direct Bilirubin",
        "unit": "mg/dL",
        "lower": 0,
        "upper": 0.3,
    },
    "blood:total_protein": {
        "name": "Total Protein",
        "unit": "g/dL",
        "lower": 6.0,
        "upper": 8.3,
    },
    "blood:c_reactive_protein": {
        "name": "CRP",
        "unit": "mg/L",
        "lower": 0,
        "upper": 10,
    },
    "blood:erythrocyte_sedimentation_rate": {
        "name": "ESR",
        "unit": "mm/hr",
        "lower": 0,
        "upper": 20,
    },
    "urine:urinalysis": {
        "name": "UA",
        "unit": "dipstick",
        "lower": 0,
        "upper": 0,
    },
    "blood:thyroid_stimulating_hormone": {
        "name": "TSH",
        "unit": "mIU/L",
        "lower": 0.4,
        "upper": 4.0,
    },
    "blood:international_normalized_ratio": {
        "name": "INR",
        "unit": "ratio",
        "lower": 0.8,
        "upper": 1.2,
    },
    "blood:prothrombin_time": {
        "name": "PT",
        "unit": "s",
        "lower": 11,
        "upper": 15,
    },
    "blood:d_dimer": {
        "name": "D-dimer",
        "unit": "µg/mL FEU",
        "lower": 0,
        "upper": 0.5,
    },
    "blood:lipase": {
        "name": "Lipase",
        "unit": "U/L",
        "lower": 23,
        "upper": 160,
    },
    "blood:amylase": {
        "name": "Amylase",
        "unit": "U/L",
        "lower": 23,
        "upper": 85,
    },
    "blood:lactate": {
        "name": "Lactate",
        "unit": "mmol/L",
        "lower": 0.5,
        "upper": 2.2,
    },
    "blood:activated_partial_thromboplastin_time": {
        "name": "aPTT",
        "unit": "s",
        "lower": 25,
        "upper": 35,
    },
    "blood:vitamin_b12": {
        "name": "B12",
        "unit": "pg/mL",
        "lower": 200,
        "upper": 900,
    },
    "blood:bilirubin_total": {
        "name": "Total Bilirubin",
        "unit": "mg/dL",
        "lower": 0.1,
        "upper": 1.2,
    },
    "blood:lactate_dehydrogenase": {
        "name": "LDH",
        "unit": "U/L",
        "lower": 140,
        "upper": 280,
    },
    "blood:creatine_kinase": {
        "name": "Creatine Kinase",
        "unit": "U/L",
        "lower": 20,
        "upper": 200,
    },
    "blood:platelet_count": {
        "name": "Platelets",
        "unit": "×10^9/L",
        "lower": 150,
        "upper": 400,
    },
    "blood:hba1c": {
        "name": "HbA1c",
        "unit": "%",
        "lower": 4.0,
        "upper": 5.6,
    },
    "blood:calcium": {
        "name": "Ca²⁺",
        "unit": "mg/dL",
        "lower": 8.5,
        "upper": 10.2,
    },
    "blood:magnesium": {
        "name": "Mg²⁺",
        "unit": "mg/dL",
        "lower": 1.7,
        "upper": 2.2,
    },
    "blood:phosphate": {
        "name": "Phosphate",
        "unit": "mg/dL",
        "lower": 2.5,
        "upper": 7,
    },
    "blood:tsh": {
        "name": "TSH",
        "unit": "mIU/L",
        "lower": 0.4,
        "upper": 4.0,
    },
    "blood:total_t3": {
        "name": "total T3",
        "unit": "ng/dL",
        "lower": 80,
        "upper": 220,
    },
    "blood:total_t4": {
        "name": "total T4",
        "unit": "mcg/dL",
        "lower": 5,
        "upper": 12,
    },
    "blood:free_t4": {
        "name": "Free T4",
        "unit": "ng/dL",
        "lower": 0.8,
        "upper": 1.8,
    },
    "blood:folate": {
        "name": "Folate",
        "unit": "ng/mL",
        "lower": 2.7,
        "upper": 17.0,
    },
    "blood:troponin_i": {
        "name": "Troponin I",
        "unit": "ng/mL",
        "lower": 0,
        "upper": 0.04,
    },
    "blood:ferritin": {
        "name": "Ferritin",
        "unit": "ng/mL",
        "lower": 12,
        "upper": 300,
    },
    "blood:urea": {
        "name": "Urea",
        "unit": "mmol/L",
        "lower": 2.5,
        "upper": 7.1,
    },
    "csf:glucose": {
        "name": "CSF Glucose",
        "unit": "mg/dL",
        "lower": 50,
        "upper": 80,
    },
    "csf:protein": {
        "name": "CSF Protein",
        "unit": "mg/dL",
        "lower": 15,
        "upper": 45,
    },
    "blood:procalcitonin": {
        "name": "Procalcitonin",
        "unit": "ng/mL",
        "lower": 0,
        "upper": 0.5,
    },
    "blood:rheumatoid_factor": {
        "name": "Rheumatoid Factor",
        "unit": "IU/mL",
        "lower": 0,
        "upper": 14,
    },
    "blood:peripheral_blood_smear": {
        "name": "Peripheral Blood Smear",
        "unit": "qualitative",
        "lower": 0,
        "upper": 0,
    },
    "blood:b_type_natriuretic_peptide": {
        "name": "BNP",
        "unit": "pg/mL",
        "lower": 0,
        "upper": 100,
    },
    "blood:reticulocyte_count": {
        "name": "Retics",
        "unit": "%",
        "lower": 0.5,
        "upper": 1.5,
    },
    "blood:partial_thromboplastin_time": {
        "name": "PTT",
        "unit": "s",
        "lower": 25,
        "upper": 35,
    },
    "blood:triglycerides": {
        "name": "TG",
        "unit": "mg/dL",
        "lower": 0,
        "upper": 150,
    },
}


def get_default_lab(key: str):
    """
    Return a random-normal InvestigationResult for the given assay key.
    """
    props = DEFAULT_VALUES.get(key)
    if not props:
        return None

    lo, hi = props["lower"], props["upper"]
    sampled = round(random.uniform(lo, hi), 1)
    return dict(
        name=props["name"],
        value=sampled,
        units=props["unit"],
        reference_range=f"{lo}–{hi} {props['unit']}",
    )
