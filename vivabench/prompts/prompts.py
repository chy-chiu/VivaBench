MEDQA_HISTORY_EXPAND_PROMPT = """You are a medical AI assistant. Given a short clinical vignette from a clinical exam, you are to return the sections about the patient history from the original input, and expand / augment the history for the patient that fits the clinical picture. Add a bit more irrelevant past medical / surgical history, medication history, family / social history as you see fit. If the patient is dead / terminal in the prompt, you need to write the patient history as if he is first presented to the hospital and still alive but unwell. Do not include any examination findings, investigation, or diagnosis in your results. Return in free text paragraph, as if it is in a SOAP clinical note. You should separate sections from the prompt with sections that are augmented. Moreover, you should not embellish existing symptoms or add too much hints that guide towards the diagnosis. 

Example input:
### INPUT START
# Exam Question: A 50 year old male came in with weight loss, haemoptysis. He died 5 months later, and autopsy showed NSCLC. What are the cell changes in the underlying disease?
Exam Answer: Squamous cell metaplasia
Diagnosis for clinical picture: Non-small cell Lung cancer
### INPUT END

NB: The Vignette Question / Answer might not be directly relevant to the diagnosis. However, you should base your expanded clinical history to the stated diagnosis.

An example output would be: 
ORIGINAL HISTORY: A 50 year old male came in with 2-week history of haemoptysis on background of 6 month weight loss. 
ADDITIONAL HISTORY: Addiitonal symptoms include some wheezing and breathing on exertion. His past medical history include COPD, reflux, hyperlipidaemia, obesity. He has a 50 pack year smoking history. His dad passed away from lung cancer when he was 5
"""

MEDQA_PHYSICAL_EXPAND_PROMPT = """You are a medical AI assistant. Given a short clinical vignette from a USMLE, you are to return the sections about physical examination of the aptient from the original input, if any, then expand / augment the examination findings to fit the patient's clinical picture. Return a paragraph of general physical examination findings with vitals that would appear in a clinical note for this patient. If the patient is dead / terminal in the prompt, you need to write the patient examination as if he is first presented to the hospital and you are examining him for the first time. Return in free text paragraph, as if it is in a SOAP clinical note. You should separate sections from the prompt with sections that are augmented. Do not include any clinical history, investigation, or diagnosis in your results.

Example input:
### INPUT START
Exam Question: A 50 year old male came in with weight loss, haemoptysis, and audible wheezing. He died 5 months later, and autopsy showed NSCLC. What are the cell changes in the underlying disease?
Exam Answer: Squamous cell metaplasia
Diagnosis for clinical picture: Non-small cell Lung cancer
### INPUT END

NB: The Vignette Question / Answer might not be directly relevant to the diagnosis. However, you should base your examination findings to the stated diagnosis.

Example output: 
ORIGINAL PHYSICAL EXAMINATION FINDINGS: Audible wheezing in the right lower lobe. 
ADDITIONAL PHYSICAL EXAMINATION FINDINGS: Other additional examination findings include: On general inspection, the patient appears cachectic with noticeable weight loss and mild respiratory distress at rest. Vital signs reveal a temperature of 37.2°C, pulse 96 bpm, respiratory rate 22 breaths per minute, blood pressure 130/80 mmHg, and oxygen saturation 92% on room air. Chest inspection reveals barrel-shaped chest, with use of accessory muscles during respiration. Palpation demonstrates decreased chest expansion bilaterally, more pronounced on the right side. Percussion over the right upper lung field is dull compared to the left, while other areas are resonant. Auscultation reveals decreased breath sounds and prolonged expiratory phase bilaterally, with coarse crackles and occasional wheezes predominantly in the right upper lobe. Cardiovascular examination shows normal S1 and S2 without murmurs, rubs, or gallops. Abdominal examination is unremarkable with no hepatosplenomegaly. Neurological and extremity exams are normal, with no clubbing or peripheral edema noted."}
"""

MEDQA_INVESTIGATION_EXPAND_PROMPT = """You are a medical AI assistant. Given a short clinical vignette from a USMLE examination question, you are to return the investigations mentioned in the clinical vignette, and expand / augment the list of investigations to fit the patient's clinical picture. Include the routine serological tests that would be done for most patients. Return a paragraph of investigation findings that would appear in a clinical note for this patient. Do not include any imaging for this patient, as we have that information separately. 

You should describe the investigation findings as if he is first presented to the hospital and untreated. You should separate sections from the prompt with sections that are augmented. Do not include any clinical history, examination findings, or diagnosis in your results.

Example input:
Exam Question: A 50 year old male came in with weight loss, haemoptysis. His hemoglobin was 8. He died 5 months later, and autopsy showed NSCLC. What are the cell changes in the underlying disease?
Exam Answer: Squamous cell metaplasia
Diagnosis for clinical picture: Non-small cell Lung cancer

NB: The Vignette Question / Answer might not be directly relevant to the diagnosis. However, you should base your examination findings to the stated diagnosis.

Example output: 
ORIGINAL INVESTIGATIONS: Bloods: Hemoglobin 8g/dL (low, mild anemia)
ADDITIONAL INVESTIGATIONS: Bloods: white blood cell count 8.5 x10^9/L (normal), platelets 320 x10^9/L (normal), sodium 138 mmol/L, potassium 4.2 mmol/L, chloride 102 mmol/L, bicarbonate 24 mmol/L, urea 6.5 mmol/L, creatinine 90 µmol/L, ALT 22 U/L, AST 28 U/L, alkaline phosphatase 85 U/L, total bilirubin 12 µmol/L, albumin 32 g/L (low), C-reactive protein (CRP) 18 mg/L (mildly elevated), prothrombin time (PT) 13 seconds (normal), INR 1.0, and lactate dehydrogenase (LDH) 280 U/L (mildly elevated).
"""

IMAGING_PARSE_EXPAND_PROMPT = """You are a medical AI assistant. Given a short clinical vignette from a USMLE examination question, you are to parse the imaging mentioned in the vignette, and potentially expand the list of imaging done to fit the patient's clinical picture as you see fit. For each generated imaging modality, return a paragraph of radiological findings that would appear in a clinical note for this patient. Do not include the diagnosis in your findings / report text. Additionally, your goal is to minimize excessive investigations. If the patient's diagnosis does not require imaging to confirm, you should not augment any imaging modalities not mentioned in the input. However, if the patient was diagnosed with / presenting with findings that wouuld have radiological findings, you should include them. Do not include any additional history, examination findings, or other investigation findings.

Return in .json format, Dict[str, ImagingResult]. Do not include any comments / calculations in your .json output. The key string should be in format "{modality} {region}" in general. The dataclass structure for ImagingResult is below for your reference:
    
class ImagingResult(ClinicalData):    
    modality: str # e.g. CT, X-Ray
    region: str # e.g. Chest, Abdomen
    report: str # Radiological findings, do not include diagnosis
    augmented: bool # Whether this was from the original prompt, or augmented

Example input 1:
Exam Question: A 50 year old male came in with weight loss, haemoptysis. His Hb was 8. He died 5 months later, and autopsy showed NSCLC. CXR showed a lung nodule. What are the cell changes in the underlying disease?
Exam Answer: Squamous cell metaplasia
Diagnosis for clinical picture: Non-small cell Lung cancer

NB: The Vignette Question / Answer might not be directly relevant to the diagnosis. However, you should base your examination findings to the stated diagnosis.

Example output 1: 
{"Chest X-Ray": 
{"modality": "X-Ray",
"region": "Chest", 
"report": "Solitary pulmonary nodule",
"augmented: true}
} 

Example input 2:
Clinical Picture: Patient with speech difficulties. CT Brain showed hyperdense lesion around the MCA
Diagnosis: Hemorrhagic Stroke

Example output 2: 
{"CT Brain": 
{"modality": "CT",
"region": "Brain", 
"report": "Hyperdense lesion around the MCA",
"augmented": false}
}

Example input 3:
Clinical Picture: Patient with a sneeze
Diagnosis: Viral infection

Example output 3: 
{} - It's a common cold! No imaging for this patient! Return an empty dictionary only. 

Example input 4: 
Clinical Picture: Patient came into hospital with a fractured rib. Ongoing monitoring showed that her hemoglobin is low. 
Diagnosis: Anaemia

Example output 4:
{"Chest X-Ray": {"modality": "X-Ray",
"region": "Chest", 
"report": "Fractured 5th rib",
"augmented: true}}

Here, even though the patient's diagnosis was anaemia, her original presentation mentioned a fractured rib. Therefore you can include a chest x-ray.
"""

INVESTIGATION_PARSE_PROMPT = """You are a medical AI assistant. Given a short clinical vignette with both original and augmented investigation results, you are to parse the investigation results to fit the patient's clinical picture. Reply in .json format, with data class format Investigations = {"bedside": Dict[str, InvestigationResult], "blood": Dict[str, InvestigationResult], "urine": Dict[str, InvestigationResult], ...}. Do not include any comments / calculations in your .json output. 

Below is the data format for Investigations and LabResult:

class Investigations(BaseModel): 
    bedside: Dict[str, Union[InvestigationResult, str]]  # For any bedside tests such as ECG
    blood: Dict[str, Union[InvestigationResult, str]]  # Any blood / serological testing
    urine: Dict[str, Union[InvestigationResult, str]]  # Any urine testing, such as urine white cell count
    csf: Dict[str, Union[InvestigationResult, str]]  # Any testing involving cerebrospinal fluid such as lumbar puncture
    other_fluid: Dict[str, Union[InvestigationResult, str]]  # Any testing involving any other extracted fluid, such as joint aspirate, ascites tap 
    microbiology: Dict[str, Union[InvestigationResult, str]]  # Any microbiology testing, such as sputum culture
    genetic: Dict[str, Union[InvestigationResult, str]]  # For genetic testing results in particular
    tissue: Dict[str, Union[InvestigationResult, str]]  # For any tissue samples, e.g. biopsy 
    other: Dict[str, str] # For any other special tests, such as lung function test. Do not include vitals here.
    
class InvestigationResult(BaseModel):
    name: str
    value: Union[str, float]
    units: Optional[str] = None
    reference_range: Optional[str] = None # If not available in prompt, no need to include
    flag: Optional[str] = None  # H, L, Critical, etc.
    note: Optional[str] = None  # e.g. location where it is sampled from, what kind of organisms, antibiotic sensitivity etc.
    specimen_type: Optional[str] = None # e.g. blood
    augmented: bool 

If an attribute is marked as Optional in the data structure, and the value is null / None, you do not need to include it in your .json return. It will be automatically filled in. 

For example, for input data: 
ORIGINAL INVESTIGATIONS: ECG normal. ALT 11 IU/L, urine WBC -ve
ADDITIONAL INVESTIGATIONS: Bloods: CRP 8

You should return:
{
    "bedside": {
        "ECG": {
            "name": "ECG", 
            "value": "normal",
            "augmented": false
        }
    }
    "blood": {
        "Alanine Aminotransferase (ALT)": {
            "name": "Alanine Aminotransferase (ALT)",
            "value": 11.0,
            "units": "IU/L",
            "augmented": false
        }, 
        "C-Reactive Protein (CRP)":  {
            "name": "C-Reactive Protein (CRP)",
            "value": 8.0,
            "augmented": true
        }
    },
    "urine": {
        "White Blood Cell": {
            "name": "White Blood Cell",
            "value": "negative",
            "augmented": false
            }
        }
}
Acceptable keys for json are components in the Investigation classes, which includes: "bedside", "blood", "urine", "csf", "other_fluid", "microbiology", "genetic", "tissue", "other"
"""


PHYSICAL_PARSE_AUG_PROMPT = """You are a medical AI assistant. Given a clinical vignette with both direct and augmented examination findings, you are to extract the physical examination for a patient for me. Do not include other information such as bloods or investigation findings. Additionally, some of the examination findings might be augmented, in which you will want to set augmented as True.  Vitals do not need the augmentation flag. 

Return a single JSON object with 'vitals' and 'systems' as top-level keys, in the format {"vitals": ...,"systems": {"respiratory": ...}}. Do not include any comments / calculations in your .json output. The data structure for your putput is included for reference:

Output Schema Pydantic:
{
    vitals: Vitals = Field(default_factory=Vitals)
    systems: Dict[str, Dict[str, PhysicalFinding]]
}

class PhysicalFinding(BaseModel):
    name: str
    description: str
    location: Optional[str] 
    severity: Optional[str] = None
    notes: Optional[str] = None
    augmented: bool # Whether this item was augmented or not

    
class Vitals(BaseModel):
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    respiratory_rate: Optional[str] = None
    oxygen_saturation: Optional[str] = None
    pain_score: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    bmi: Optional[float] = None

If an attribute is marked as Optional in the data structure, and the value is null / None, you do not need to include it in your .json return. It will be automatically filled in. 

for "systems", acceptable headers include:
["general", "peripheral", "cardiovascular", "respiratory", "HEENT", "gastrointestinal", "genitourinary", "endocrine", "neurological", "psychiatric", "musculoskeletal", "dermatological"]

For example, for input:
ORIGINAL PHYSICAL EXAMINATION: 49 male, morbidly obese, wheeze and crackles in the right lower lobe upon auscultation, BMI 45, BP 160/110
ADDITIONAL PHYSICAL EXAMINATION: Patient has clubbing in his fingers

You should return:
{
  "vitals": {
    "blood_pressure_systolic": 160,
    "blood_pressure_diastolic": 110,
    "bmi": 45
  },
  "systems": {
    "respiratory": {
      "wheezing": {
        "name": "wheezing",
        "description": "wheeze upon auscultation",
        "location": "right lower lobe",
        "augmented": false
      },
      "crackles": {
        "name": "crackles",
        "description": "crackles upon auscultation",
        "location": "right lower lobe",
        "augmented": false
      }
    },
    "peripheral": {
      "clubbing": {
        "name": "clubbing",
        "description": "clubbing in his fingers",
        "augmented": true
      }
    }
  }
}
"""

HISTORY_PARSE_AUGMENTED = """
You are a medical AI assistant. Given a clinical vignette, you are to extract the demographics and history for a patient for me. Do not include other information such as bloods, physical examination, or investigation findings. If the patient is dead / terminal in the prompt, you need to process the patient history as if he is first presented to the hospital and still able to present a history. Do not include information about death or terminal status in the history fields; focus on the presenting history. Return everything in a structured format as per the dataclass structure below, except for the symptom list, which is in free text. Some of the symptoms might be marked as augmented, and you need to delineate that in your list of symptoms. Anything that is not a symptom we do not care about augmentation or lack thereof. The chief complaint should be the most urgent symptom(s) from the ORIGINAL history only, not including augmented symptoms.

Return a single JSON object, with no comments or explanations, with format {"demographics": Demographics, "history": History}. Do not include any comments / calculations in your .json output. Follow data structure below: 

class Demographics(BaseModel):
    age: Union[int, str] # For ages < 1, input number of weeks / months etc.
    gender: str
    race: Optional[str] = None
    ethnicity: Optional[str] = None
    place_of_birth: Optional[str] = None

class History(BaseModel):
    chief_complaint: str # This should be a single phrase, with only the most urgent symptoms. 
    history_of_present_illness: str # This is the full course of the disease, if relevant. Combine both original and augmented history.
    hopc_structured: Dict[str, str] # This is a structured way to organise any information that could be useful for diagnosis, that is not a symptom. For example, any recent medication changes, exposure to sick people etc. Only include clues from the original history in this field as appropriate.
    symptoms_freetext: str # Full list of symptoms the patient is experiencing, separated by original and augmented sections. 
    past_medical_history: List[str] = Field(default_factory=list) # Full list of past medical history, as strings
    medications: List[Medication] = None
    allergies: List[Allergy] = None
    social_history: Optional[SocialHistory] = None
    family_history: List[FamilyHistory] = Field(default_factory=list)

class Medication(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    indication: Optional[str] = None
    current: bool = true
    
class Allergy(BaseModel):
    allergen: str
    reaction: Optional[str] = None
    severity: Optional[str] = None

class SocialHistory(BaseModel):
    smoking_current: Optional[str] = None
    smoking_pack_years: Optional[float] = None
    alcohol_use: Optional[str] = None
    substance_use: Optional[str] = None
    occupation: Optional[str] = None
    living_situation: Optional[str] = None
    travel_history: Optional[str] = None
    exercise: Optional[str] = None
    diet: Optional[str] = None
    sexual: Optional[str] = None
    other: Optional[Dict[str, str]] = None
    
class FamilyHistory(BaseModel):
    condition: str
    relationship: str
    age_at_onset: Optional[int] = None
    notes: Optional[str] = None
    
If there are no family history, medications, or allergies, these fields may be omitted or set to null.
If an attribute is marked as Optional in the data structure, and the value is null / None, you do not need to include it in your .json return. It will be automatically filled in. 
    
For example, for input:
ORIGINAL HISTORY: 63 year old man with 2 hour history of nausea after eating a bad sandwich. PMH includes T2DM and hypertension.  He recently also started a new medication that could lead to nausea
ADDITIONAL HISTORY: Patient also has watery diarrhoea. PMH include T2DM, HTN, PAD.

You should return:
{"demographics": {"age": 63, "gender": "male"},
 "history": {"chief_complaint": "Nausea",
             "history_of_present_illness": "2 hour history of nausea after eating a bad sandwich.",
             "hopc_structured": {"food_consumption": "Ate a sandwich that smelled a bit funny", "medication_change": "Recently changed his medication that gives him nausea."}
             "symptoms_freetext": "ORIGINAL SYMPTOMS: 63 year old man with 2 hour history of nausea after eating a bad sandwich. ADDITIONAL HISTORY: Patient also has watery diarrhoea",      
            "past_medical_history": [
            "Type 2 diabetes mellitus",
            "Hypertension",
            "peripheral arterial disease"]}} 
            
Noting that things such as past medical history don't require an augmentation flag. 
Reminder again, ensure your return is purely .json, and does not include any comments or calculations within your output.

ILLEGAL EXAMPLE: 
 "social_history": {
      "smoking_pack_years": 0.5 * 20 / 1, 
}
"""

SYMPTOMS_PARSE_AUGMENTED = """
You are a medical AI assistant. Given a list of patient symptoms in free text, you are to process them in a structured manner. Additionally, some of the symptoms might be augmented, in which you will want to set augmented as True.
        
Return your output as Dict[str, Symptom]. 

The dataclass structure for Symptom is as below: 

class Symptom(BaseModel):
    name: str
    system: str # Which system these symptoms belong to
    severity: Optional[str] = None
    onset: Optional[str] = None  # sudden, gradual
    duration: Optional[str] = None  # e.g., "2 days", "3 weeks"
    location: Optional[str] = None
    character: Optional[str] = None
    radiation: Optional[str] = None
    alleviating_factors: List[str] = Field(default_factory=list)
    aggravating_factors: List[str] = Field(default_factory=list)
    associated_symptoms: List[str] = Field(default_factory=list)
    timing: Optional[str] = None  # constant, intermittent, etc.
    context: Optional[str] = None  # circumstances when symptom occurs
    notes: Optional[str] = None
    augmented: bool = False

If an attribute is marked as Optional in the data structure, and the value is null / None, you do not need to include it in your .json return. It will be automatically filled in. 

For "system", acceptable headers include:
["general", "cardiovascular", "respiratory", "HEENT", "gastrointestinal", "genitourinary", "endocrine", "neurological", "psychiatric", "musculoskeletal", "dermatological"]

For example, for input:
ORIGINAL SYMPTOMS: 63 year old man with 2 hour history of nausea after eating a bad sandwich
ADDITIONAL HISTORY: Patient also has watery diarrhoea

You should return:
{
            "Nausea": {
                "name": "Nausea",
                "system": "gastrointestinal",
                "onset": "acute",
                "duration": "2 hours",
                "augmented": false
            },
            "Diarrhoea": {
                "name": "Diarrhoea",
                "system": "gastrointestinal",
                "character": "watery",
                "augmented": true
            }
}
"""


HISTORY_PARSE_PROMPT = """
You are a medical AI assistant. Given a clinical vignette, you are to extract the demographics and history for a patient for me. Do not include other information such as bloods, physical examination, or investigation findings. You are to parse the patient history as if the patient is first presented to the hospital. If the patient is dead / terminal in the prompt, you need to process the patient history as if the patient is still able to present a history. Do not include information about death or terminal status in the history fields; focus on the presenting history. Return everything in a structured format as per the dataclass structure below, except for the symptom list, which is in free text. The chief complaint a single phrase, consisting of the most urgent symptom(s) from the source history only, and it should be non-specific to diagnosis. You should use standardised terms whenever you can, and not any short hand that doctors commonly use. Your main goal is to be as high fidelity in semantic meaning to the input as possible, and you must avoid including any unwanted or untrue information at all costs, err on the side of caution. 

Return a single JSON object, with no comments or explanations, with format {"demographics": Demographics, "history": History}. Do not include any comments / calculations in your .json output. Follow data structure below: 

class Demographics(BaseModel):
    age: Union[int, str] # For ages < 1, input number of weeks / months etc.
    gender: str
    race: Optional[str] = None
    ethnicity: Optional[str] = None
    place_of_birth: Optional[str] = None

class History(BaseModel):
    chief_complaint: str # This should be a single phrase, with only the most urgent symptoms. 
    history_of_present_illness: str # This is the full course of the disease, if relevant. 
    hopc_structured: Dict[str, str] # This is a structured way to organise any information that could be useful for diagnosis, that is not a symptom. For example, any recent medication changes, exposure to sick people etc
    symptoms_freetext: str # Full list of phrases on the symptoms the patient is experiencing. It should be copied verbatim from the source, optimally with shorthand replaced, but semantic meaning preserved. 
    past_medical_history: List[str] = Field(default_factory=list) # Full list of past medical history, as strings. Again, should be copied verbatim from source when possible. 
    medications: List[Medication] = None # List of medications patient is taking. See the structure for medication below. 
    allergies: List[Allergy] = None # List of allergies the patient might have. See the structure for medication below. 
    social_history: Optional[SocialHistory] = None # Social history for the patient. See the structure below. 
    family_history: List[FamilyHistory] = Field(default_factory=list) # Family history for the patient. 

class Medication(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    indication: Optional[str] = None
    current: bool = true
    
class Allergy(BaseModel):
    allergen: str
    reaction: Optional[str] = None
    severity: Optional[str] = None

class SocialHistory(BaseModel):
    smoking_current: Optional[str] = None
    smoking_pack_years: Optional[float] = None
    alcohol_use: Optional[str] = None
    substance_use: Optional[str] = None
    occupation: Optional[str] = None
    living_situation: Optional[str] = None
    travel_history: Optional[str] = None
    exercise: Optional[str] = None
    diet: Optional[str] = None
    sexual: Optional[str] = None
    other: Optional[Dict[str, str]] = None
    
class FamilyHistory(BaseModel):
    condition: str
    relationship: str
    age_at_onset: Optional[int] = None
    notes: Optional[str] = None
    
If there are no family history, medications, or allergies, these fields may be omitted or set to null.
If an attribute is marked as Optional in the data structure, and the value is null / None, you do not need to include it in your .json return. It will be automatically filled in. 

Example input:     
63 year old man with 5 hour history of nausea after eating a sandwich that smelled a bit funny. PMH includes T2DM and hypertension. He recently also started a new medication that could lead to nausea. Patient also has also been experiencing watery diarrhoea and vomiting the last 2 hours. PMH include T2DM, HTN, PAD.

You should return:
{"demographics": {"age": 63, "gender": "male"},
 "history": {"chief_complaint": "Nausea",
             "history_of_present_illness": "2 hour history of nausea after eating a bad sandwich.",
             "hopc_structured": {"food_consumption": "Ate a sandwich that smelled a bit funny", "medication_change": "Recently changed his medication that gives him nausea."}
             "symptoms_freetext": "2 hour history of nausea after eating a sandwich that smelled a bit funny. Patient also has also been experiencing watery diarrhoea and vomiting the last 2 hours",      
            "past_medical_history": [
            "Type 2 diabetes mellitus",
            "Hypertension",
            "peripheral arterial disease"]}} 
            
Reminder, ensure your return is purely .json, and does not include any comments or calculations within your output.

ILLEGAL EXAMPLE: 
 "social_history": {
      "smoking_pack_years": 0.5 * 20 / 1, 
}
"""


PHYSICAL_PARSE_PROMPT = """You are a medical AI assistant. Given a clinical vignette with examination findings, you are to extract the physical examination for a patient for me. Do not include other information such as bloods or investigation findings. 

Return a single JSON object with 'vitals' and 'systems' as top-level keys, in the format {"vitals": ...,"systems": {"respiratory": ...}}. Do not include any comments / calculations in your .json output. The data structure for your putput is included for reference:

Output Schema Pydantic:
{
    vitals: Vitals = Field(default_factory=Vitals)
    systems: Dict[str, Dict[str, PhysicalFinding]]
}

class PhysicalFinding(BaseModel):
    name: str
    description: str
    location: Optional[str] 
    severity: Optional[str] = None
    notes: Optional[str] = None
    
class Vitals(BaseModel):
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    respiratory_rate: Optional[str] = None
    oxygen_saturation: Optional[str] = None
    pain_score: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    bmi: Optional[float] = None

If an attribute is marked as Optional in the data structure, and the value is null / None, you do not need to include it in your .json return. It will be automatically filled in. 

for "systems", acceptable headers include:
["general", "peripheral", "cardiovascular", "respiratory", "HEENT", "gastrointestinal", "genitourinary", "endocrine", "neurological", "psychiatric", "musculoskeletal", "dermatological"]

For example, for input:
49 male, with a 45 pack year history. Morbidly obese. wheeze and crackles in the right lower lobe upon auscultation, BMI 45, BP 160/110

You should return:
{
  "vitals": {
    "blood_pressure_systolic": 160,
    "blood_pressure_diastolic": 110,
    "bmi": 45
  },
  "systems": {
    "respiratory": {
      "wheezing": {
        "name": "wheezing",
        "description": "wheeze upon auscultation",
        "location": "right lower lobe",
      },
      "crackles": {
        "name": "crackles",
        "description": "crackles upon auscultation",
        "location": "right lower lobe",
      }
    },
  }
}
"""

SYMPTOMS_PARSE_PROMPT = """You are a medical AI assistant. Given a list of patient symptoms in free text, you are to group them in a structured manner. Return your output as Dict[str, Symptom]. The dataclass structure for Symptom is as below: 

class Symptom(BaseModel):
    name: str
    system: str # Which system these symptoms belong to
    severity: Optional[str] = None
    onset: Optional[str] = None  # sudden, gradual
    duration: Optional[str] = None  # e.g., "2 days", "3 weeks"
    location: Optional[str] = None
    character: Optional[str] = None
    radiation: Optional[str] = None
    alleviating_factors: List[str] = Field(default_factory=list)
    aggravating_factors: List[str] = Field(default_factory=list)
    associated_symptoms: List[str] = Field(default_factory=list)
    timing: Optional[str] = None  # constant, intermittent, etc.
    context: Optional[str] = None  # circumstances when symptom occurs
    notes: Optional[str] = None
    
If an attribute is marked as Optional in the data structure, and the value is null / None, you do not need to include it in your .json return. It will be automatically filled in. 

For "system", acceptable headers include:
["general", "cardiovascular", "respiratory", "HEENT", "gastrointestinal", "genitourinary", "endocrine", "neurological", "psychiatric", "musculoskeletal", "dermatological"]

For example, for input:
63 year old man with 2 hour history of nausea after eating a bad sandwich. Patient also has watery diarrhoea

You should return:
{
            "Nausea": {
                "name": "Nausea",
                "system": "gastrointestinal",
                "onset": "acute",
                "duration": "2 hours",
            },
            "Diarrhoea": {
                "name": "Diarrhoea",
                "system": "gastrointestinal",
                "character": "watery",
            }

Use SNOMED official names for any symptoms / findings provided below:
['Encopresis', 'Bradycardia', 'Encopresis with constipation AND overflow incontinence', 'Nocturnal enuresis', 'Easy bruising', 'Unexplained weight loss', 'Cramp in limb', 'Unresponsive', 'Mass lesion of brain', 'Noncompliance with treatment', 'Uninsured medical expenses', 'Nausea', 'Unbalanced diet', 'Complicated grieving', 'Active advance directive (copy within chart)', 'Active living will', 'Pain in female pelvis', 'Old healed fracture of bone', 'Mass of pancreas', 'Solitary nodule of lung', 'Instability of femoropatellar joint', 'Mass of urinary bladder', 'Stuttering', 'Dependence on hemodialysis due to end stage renal disease', 'Mass of shoulder region', 'Dependence on continuous positive airway pressure ventilation', 'Localized superficial swelling of skin', 'Chronic pain due to injury', 'Mass of head', 'Viremia', 'Body mass index 25-29 - overweight', 'Gravid uterus size for dates discrepancy', 'Chronic pain in face', 'Bacteremia', 'Pain in male perineum', 'Mass of pituitary', 'Atypical squamous cells of undetermined significance on cervical Papanicolaou smear', 'Thallium stress test abnormal', 'Mass of thyroid gland', 'Mass of mediastinum', 'Mass of retroperitoneal structure', 'Mass of testicle', 'Mass of chest wall', 'Atypical squamous cells on cervical Papanicolaou smear cannot exclude high grade squamous intraepithelial lesion', 'Carrier of vancomycin resistant enterococcus', 'Mass of adrenal gland', 'Mass of pelvic structure', 'Subcutaneous nodule', 'Mass of thoracic structure', 'Atypical squamous cells of undetermined significance on vaginal Papanicolaou smear', 'Abnormal cervical Papanicolaou smear', 'Mass of scrotum', 'Cardiovascular stress test abnormal', 'Imaging of lung abnormal', 'Mass of tongue', 'Gestational age unknown', 'Early satiety', 'Electrocardiogram abnormal', 'Pelvic swelling', 'Emotional stress', 'Nonspecific tuberculin test reaction', 'Computed tomography result abnormal', 'Mass of foot', 'Drug seeking behavior', 'Cardiac defibrillator in situ', 'Cramp in lower leg associated with rest', 'Cardiac pacemaker in situ', 'Abnormal gait', 'Periumbilical pain', 'Willing to be donor of liver', 'Decorative tattoo of skin', 'Callus of bone', 'Willing to be donor of kidney', 'Requires a tetanus booster', 'Multigravida of advanced maternal age', 'Immunoglobulin G subclass deficiency', 'High risk pregnancy due to history of preterm labor', 'Male urinary stress incontinence', 'Fussy toddler', 'Pain in forearm', 'Mass of submandibular region', 'Irregular bowel habits', 'Weakness of vocal cord', 'Chronic pain in female pelvis', 'Low density lipoprotein cholesterol above reference range', 'Periodic leg movements of sleep', 'Clotting time above reference range', 'Low lying placenta', 'Anovulatory amenorrhea', 'Sensory ataxia', 'Antinuclear antibody above reference range', 'Fibrocystic breast changes', 'Helicobacter pylori antibody above reference range', 'Dependence on ventilator', 'Orthostatic headache', 'Edema of face', 'Carcinoembryonic antigen above reference range', 'Nonspecific syndrome suggestive of viral illness', 'Nocturia', 'Edema of lower leg', 'Swelling of upper arm', 'Unsatisfactory cardiotochogram tracing', 'Cramp in lower leg', 'False labor', 'Bowing deformity of lower limb', 'Red eye', 'Digestive system reflux', 'Not up to date with immunizations', 'Physical deconditioning', 'Muscle weakness of limb', 'Magnetic resonance imaging scan abnormal', 'Short-sleeper', 'Requires vaccination', 'Swelling of bilateral lower limbs', 'Intermittent claudication', 'Diverticulosis of colon without diverticulitis', 'Diverticulosis of sigmoid colon', 'Tenderness of temporomandibular joint', 'Constantly crying infant', 'Constipation', 'Allergy to peanut', 'Calcaneal spur', 'Hyperemia of eye', 'Allergy to soy protein', 'Allergy to dust mite protein', 'Non-healing surgical wound', 'Allergy to drug', 'Intolerance to food', 'Environmental allergy', 'Allergy to food', 'Excessive self-criticism', 'Allergic disposition', 'Neuralgia', 'Sacroiliac instability', 'Localized swelling of abdominal wall', 'Allergy to penicillin', 'Lax vaginal introitus', 'Has special educational needs', 'Sexually assaultive behavior', 'Peripheral neuralgia', 'Drusen of optic disc', 'Nasal discharge', 'Intercostal neuralgia', 'Teething syndrome', 'Hypovolemia', 'Sebaceous hyperplasia', 'Pain of joint of knee', 'Skin irritation', 'Mass of head and/or neck', 'Pain of knee region', 'Bloodstained liquor', 'Bloodstained sputum', 'Blurring of visual image', 'Body mass index 30+ - obesity', 'Body weight problem', 'Bone pain', 'Borderline blood pressure', 'Macrocephaly', 'Syncope and collapse', 'Breast fed', 'Breast finding', 'Breast lump', 'Breasts asymmetrical', 'Breath smells unpleasant', 'Breech presentation', 'Elevated level of transaminase and lactic acid dehydrogenase', 'Abdominal bloating', 'Abdominal bruit', 'Excessive weight gain measured during pregnancy', 'Abdominal colic', 'Abdominal discomfort', 'Bronchospasm', 'Syncope', 'Abdominal mass', 'Bruit', 'Abdominal pain', 'Abdominal pain in pregnancy', 'Vasovagal syncope', 'Microcephaly', 'Abdominal wall pain', 'Burning sensation', 'Burning sensation in eye', 'Burping', 'Cachexia', 'Cardiac syndrome X', 'Cervicogenic headache', 'Chest discomfort', 'Chest pain', 'Chest pain on exertion', 'Chest swelling', 'Chest wall pain', 'Chews tobacco', 'Childhood growth AND/OR development alteration', 'Chill', 'Choking', 'Cholestasis', 'Chronic abdominal pain', 'Chronic anxiety', 'Chronic cough', 'Chronic pain', 'Abnormal biochemical finding on antenatal screening of mother', 'Abnormal blood pressure', 'Abnormal cervical smear', 'Cigarette smoker', 'Abnormal defecation', 'Abnormal deglutition', 'Claustrophobia', 'Abnormal female sexual function', 'Clearing throat - hawking', 'Clicking hip', 'Abnormal findings on diagnostic imaging of lung', 'Abnormal liver function', 'Abnormal male sexual function', 'Abnormal posture', 'Clouded consciousness', 'Coagulation/bleeding tests abnormal', 'Abnormal renal function', 'Abnormal sexual function', 'Coin lesion of lung', 'Abnormal sputum', 'Abnormal urine', 'Abnormal vaginal bleeding', 'Abnormal vision', 'Abnormal voice', 'Colostomy present', 'Abnormal weight gain', 'Abnormal weight loss', 'Atypical absence seizure', 'Apnea in newborn', 'Absence seizure', 'Pain in coccyx', 'Pregnancy', 'Generalized edema', 'Mixed urinary incontinence', 'Coordination problem', 'Absence of breast', 'Cough', 'Cramp', 'Academic underachievement', 'Current drinker of alcohol', 'Cyanosis', 'Decrease in height', 'Decreased estrogen level', 'Decreased hearing', 'Decreased muscle tone', 'Decreased range of cervical spine movement', 'Decreased range of knee movement', 'Defective dental restoration', 'Deformity of foot', 'Deformity of hand', 'Deformity of hip joint', 'Deformity of knee joint', 'Delay when starting to pass urine', 'Delayed articulatory and language development', 'Delayed milestone', 'Deliveries by cesarean', 'Diarrhea', 'Diastolic dysfunction', 'Difficulty sleeping', 'Difficulty swallowing', 'Difficulty talking', 'Discharge from penis', 'Discoloration of skin', 'Distorted body image', 'Disturbance in sleep behavior', 'Disturbance in speech', 'Dizziness and giddiness', 'Does use hearing aid', 'Dribbling of urine', 'Drowsy', 'Dysarthria', 'Dysesthesia', 'Dysfunctional voiding of urine', 'Dyskinesia', 'Dysphasia', 'Dyspnea', 'Dyspnea on exertion', 'Dysuria', 'Ear pressure sensation', 'Ear problem', 'Ecchymosis', 'Echocardiogram abnormal', 'Edema', 'Edema of foot', 'Edema of lower extremity', 'Edema of the upper extremity', 'Edentulous', 'Education and/or schooling finding', 'Educational problem', 'Elbow joint pain', 'Elbow joint unstable', 'Elderly primigravida', 'Electroencephalogram abnormal', 'Employment problem', 'Enlarged uterus', 'Epigastric pain', 'Epileptic seizure', 'Erythema', 'Ex-smoker', 'Excess skin of eyelid', 'Excessive and frequent menstruation', 'Excessive sweating', 'Excessive thirst', 'Excessive upper gastrointestinal gas', 'Exercise tolerance test abnormal', 'Facet joint pain', 'Facial spasm', 'Facial swelling', 'Failure to gain weight', 'Failure to progress in second stage of labor', 'Falls', 'Acromioclavicular joint pain', 'Family disruption', 'Family problems', 'Family tension', 'Fatigue', 'Fatty stool', 'Fear of becoming fat', 'Febrile convulsion', 'Feces contents abnormal', 'Feeding difficulties and mismanagement', 'Feeding poor', 'Feeding problem', 'Feeding problems in newborn', 'Feeling agitated', 'Feeling angry', 'Feeling irritable', 'Feeling of lump in throat', 'Feeling suicidal', 'Female urinary stress incontinence', 'Fetal heart rate absent', 'Fever', 'Active range of joint movement reduced', 'Financial problem', 'Activity intolerance', 'Fine motor impairment', 'First stage of labor', 'Flank pain', 'Flatulence, eructation and gas pain', 'Follow-up orthopedic assessment', 'Foot joint pain', 'Foot pain', 'Foot-drop', 'Footling breech presentation', 'Frontal headache', 'Functional heart murmur', 'Funny turn', 'Pain of ear', 'Relationship problem', 'Gastrostomy present', 'Mammographic mass of breast', 'General health deterioration', 'Generalized abdominal pain', 'Generalized aches and pains', 'Generalized pruritus', 'Generally unwell', 'Unsettled infant', 'Genuine stress incontinence', 'Acute pain', 'Congenital anteversion of femur', 'Glycosuria', 'Good neonatal condition at birth', 'Groin mass', 'Gross motor impairment', 'Habitual drinker', 'Hallucinations', 'Hand joint pain', 'Hand joint stiff', 'Hand pain', 'Head tilt', 'Headache', 'Hearing problem', 'Heart murmur', 'Heartburn', 'Heavy drinker', 'Heel pain', 'Hemianopia', 'Hemoptysis', 'Hemospermia', 'Hepatitis A immune', 'Hepatitis B carrier', 'Hepatitis B immune', 'Hepatitis C carrier', 'Administrative reason for encounter', 'Hiccoughs', 'High risk pregnancy', 'High risk sexual behavior', 'Hip pain', 'Hip stiff', 'Hoarse', 'Homeless', 'Homonymous hemianopia', 'Housing lack', 'Housing unsatisfactory', 'Hyperactive behavior', 'Hypercoagulability state', 'Hyperreflexia', 'Hyperventilation', 'Hypesthesia', 'Hypogammaglobulinemia', 'Hypothermia', 'Ileostomy present', 'Impaired cognition', 'Impaired mobility', 'Impairment of balance', 'Inattention', 'Incomplete placenta at delivery', 'Incontinence', 'Incontinence of feces', 'Increased frequency of urination', 'Indigestion', 'Ineffective infant feeding pattern', 'Ineffective thermoregulation', 'Infantile colic', 'Infertile', 'Influenza-like illness', 'Inguinal pain', 'Intellectual functioning disability', 'Intention tremor', 'Intolerant of cold', 'Intolerant of heat', 'Intrauterine pregnancy', 'Glucose tolerance test outside reference range', 'Generalized onset epileptic seizure', 'Lipid above reference range', 'Blood chemistry outside reference range', 'Serum cholesterol within reference range', 'Serum iron above reference range', 'Serum creatinine above reference range', 'Alkaline phosphatase above reference range', 'Aspartate aminotransferase serum level above reference range', 'Cancer antigen 125 above reference range', 'Uses contraception', 'Serum calcium level above reference range', 'Creatine kinase level above reference range', 'C-reactive protein outside reference range', 'Tonic-clonic epileptic seizure', 'Uses depot contraception', 'Serum cholesterol above reference range', 'Irregular heart beat', 'Irregular periods', 'Blood glucose outside reference range', 'Testosterone level below reference range', 'Prostate specific antigen outside reference range', 'Thyroid stimulating hormone level above reference range', 'Generalized onset tonic-clonic epileptic seizure', 'Erythrocyte sedimentation rate above reference range', 'Renal function tests outside reference range', 'Alanine aminotransferase above reference range', 'Age-related cognitive decline', 'Serum ferritin above reference range', 'Prostate specific antigen above reference range', 'Liver enzymes outside reference range', 'Uses intrauterine device contraception', 'Eosinophil count above reference range', 'Uses oral contraception', 'Liver function tests outside reference range', 'Lipids outside reference range', 'Jaundice', 'Jaw pain', 'Joint pain', 'Aggressive behavior', 'Joint swelling', 'Knee stiff', 'Large prostate', 'Laryngismus', 'Learning difficulties', 'Albuminuria', 'Left lower quadrant pain', 'Left sided abdominal pain', 'Left upper quadrant pain', 'Legal problem', 'Lethargy', 'Light cigarette smoker (1-9 cigs/day)', 'Lightheadedness', 'Livebirth', 'Liver mass', 'Lives alone', 'Living in residential institution', 'Localized edema', 'Localized pain', 'Loin pain', 'Long-term drug misuser', 'Loss of appetite', 'Loss of part of visual field', 'Loss of sense of smell', 'Loss of voice', 'Low maternal weight gain', 'Lower abdominal pain', 'Lower urinary tract symptoms', 'Lump on face', 'Lump on finger', 'Lung field abnormal', 'Lung mass', 'Macrocytosis, red cells', 'Macular drusen', 'At increased risk for infection', 'At increased risk for impaired skin integrity', 'Cytomegalovirus antibody detected in serum', 'Cytomegalovirus antibody not detected in serum', 'Abnormal cervical Papanicolaou smear with human papillomavirus deoxyribonucleic acid detected', 'Multiple pregnancy', 'Celiac disease detected by autoantibody screening', 'At increased risk for deliberate self harm', 'Redness of throat', 'Autoantibody titer detected', 'Malaise', 'Malaise and fatigue', 'At increased risk for noncompliance', 'Syphilis titer detected', 'Excessive eating', 'Twin pregnancy', 'Pseudophakic intraocular lens present', 'At increased risk of sexually transmitted infection', 'At increased risk for falls', 'Rheumatoid factor detected', 'Human immunodeficiency virus detected', 'Sore throat', 'At increased risk of coronary heart disease', 'Anti-nuclear factor detected', 'Epstein-Barr virus antibody detected in serum', 'Occult blood detected in feces', 'Malingering', 'Mammography abnormal', 'Mammography normal', 'Mantoux: negative', 'Mantoux: positive', 'Marital problems', 'Mass of axilla', 'Mass of body structure', 'Mass of lower limb', 'Mass of neck', 'Mass of ovary', 'Mass of parotid gland', 'Mass of skin', 'Mass of vulva', 'Alteration in nutrition: less than body requirements', 'Altered bowel function', 'Memory impairment', 'Menometrorrhagia', 'Menopausal flushing', 'Menopausal problem', 'Menopausal symptom', 'Menopause present', 'Menorrhagia', 'Alveolar hypoventilation', 'Metatarsalgia', 'Hyponatremia', 'Focal to bilateral tonic-clonic epileptic seizure', 'Hypervolemia', 'Focal onset aware epileptic seizure', 'Microalbuminuria', 'Focal onset epileptic seizure', 'Microcytosis, red cells', 'International normalized ratio above reference range', 'Body fluid retention', 'Focal onset impaired awareness epileptic seizure', 'Mild memory disturbance', 'Amenorrhea', 'Moderate smoker (20 or less per day)', 'Mood swings', 'Multigravida', 'Multiparous', 'Multiple bruising', 'Multiple joint pain', 'Muscle fasciculation', 'Muscle pain', 'Muscle weakness', 'Musculoskeletal chest pain', 'Musculoskeletal pain', 'Myoclonus', 'Amnesia', 'Amniotic fluid -meconium stain', 'Narcotic drug user', 'Nasal congestion', 'Nasal deviation', 'Nasal sinus problem', 'Neck pain', 'Neck swelling', 'Needs influenza immunization', 'Amputated above knee', 'Amputated big toe', 'Neurogenic claudication', 'Neuropathic pain', 'Neutropenia', 'Never smoked tobacco', 'Amputee', 'Night sweats', 'No abnormality detected', 'No liquor observed vaginally', 'Nocturnal muscle spasm', 'Non-cardiac chest pain', 'Non-smoker', 'Noncompliance with diagnostic testing', 'Noncompliance with dietary regimen', 'Noncompliance with medication regimen', 'Noncompliance with therapeutic regimen', 'Normal labor', 'Normal menstrual cycle', 'Normal pregnancy', 'Not for resuscitation', 'Not yet walking', 'Numbness', 'Numbness of face', 'Numbness of foot', 'Numbness of hand', 'Numbness of lower limb', 'Objective tinnitus', 'Occipital headache', 'Occipitoanterior position', 'Oligomenorrhea', 'Anergy', 'Neonatal jaundice', 'Orthopnea', 'Neonatal jaundice associated with preterm delivery', 'Dysfunction of urinary bladder', 'Neonatal seizure', 'Spastic neurogenic urinary bladder', 'Neonatal jaundice due to glucose-6-phosphate dehydrogenase deficiency', 'Spasm of urinary bladder', 'Flaccid neurogenic urinary bladder', 'Pain of urinary bladder', 'Neurogenic urinary bladder', 'Neonatal jaundice due to delayed conjugation from breast milk inhibitor', 'Incomplete emptying of urinary bladder', 'Newborn physiological jaundice', 'Inactive tuberculosis', 'Neonatal jaundice due to delayed conjugation', 'Overweight', 'Pain', 'Pain in axilla', 'Pain in buttock', 'Pain in calf', 'Pain in cervical spine', 'Pain in elbow', 'Pain in eye', 'Pain in face', 'Pain in female genitalia', 'Pain in female genitalia on intercourse', 'Pain in finger', 'Pain in limb', 'Pain in lower limb', 'Pain in pelvis', 'Pain in penis', 'Pain in scrotum', 'Pain in testicle', 'Pain in thoracic spine', 'Pain in thumb', 'Pain in toe', 'Pain in upper limb', 'Pain in wrist', 'Pain of breast', 'Pain of sternum', 'Painful mouth', 'Ankle edema', 'Palpitations', 'Ankle instability', 'Ankle joint pain', 'Panic attack', 'Ankle pain', 'Paralysis', 'Parent-child problem', 'Parental anxiety', 'Parental concern about child', 'Paresis of lower extremity', 'Paresthesia', 'Paresthesia of foot', 'Paresthesia of hand', 'Partnership problems', 'Passive smoker', 'Patient post percutaneous transluminal coronary angioplasty', 'Anorectal pain', 'Anovulation', 'Perineal pain', 'Antenatal ultrasound scan abnormal', 'Peripheral visual field defect', 'Personal care impairment', 'Photosensitivity', 'Pins and needles', 'Pleuritic pain', 'Poly-drug misuser', 'Polypharmacy', 'Polyuria', 'Poor short-term memory', 'Poor sleep pattern', 'Post-micturition incontinence', 'Postcoital bleeding', 'Postmature infancy', 'Postmenopausal bleeding', 'Postmenopausal state', 'Postoperative pain', 'Postoperative visit', 'Postpartum state', 'Posttraumatic headache', 'Precordial pain', 'Pregnancy test negative', 'Pregnancy test positive', 'Pregnancy with abnormal glucose tolerance test', 'Pregnant - planned', 'Premature birth of newborn', 'Premature delivery', 'Premature ejaculation', 'Premature infant', 'Premature labor', 'Premature menopause', 'Inflammation of joint of wrist', 'Inflammation of sacroiliac joint', 'Inflammation of joint of hip', 'Abnormal finding on antenatal screening of mother', 'Carrier of methicillin resistant Staphylococcus aureus', 'Xerostomia', 'Plain X-ray of chest abnormal', 'Inflammation of shoulder joint', 'Inflammation of joint of finger', 'Inflammation of joint of hand', 'Tonic-clonic status epilepticus', 'Primigravida', 'Inflammation of joint of foot', 'Delusion', 'Lytic lesion of bone on plain X-ray', 'Inflammation of joint of ankle', 'Problem behavior', 'Problem situation relating to social and personal history', 'Problematic behavior in children', 'Productive cough', 'Prolonged QT interval', 'Prostate mass', 'Proteinuria', 'Proximal muscle weakness', 'Psychalgia', 'Psychosexual dysfunction', 'Ptosis of eyebrow', 'Pulmonary aspiration', 'Anxiety', 'Anxiety attack', 'Anxiety state', 'Pyrexia of unknown origin', 'Pyuria', 'Radiology result abnormal', 'Range of joint movement increased', 'Aphasia', 'Rectal mass', 'Rectal pain', 'Recurrent falls', 'Reduced fetal movement', 'Reduced libido', 'Reduced visual acuity', 'Apnea', 'Regional lymph node metastasis present', 'Renal colic', 'Renal mass', 'Repeat prescription card duplicate issue', 'Requires a meningitis vaccination', 'Requires polio vaccination', 'Respiratory crackles', 'Respiratory distress', 'Resting tremor', 'Retinal drusen', 'Apraxia', 'Retrosternal pain', 'Rib pain', 'Right lower quadrant pain', 'Right upper quadrant pain', 'Rubella non-immune', 'Rubella status not known', 'Sacral dimple', 'Sacroiliac joint pain', 'Arthralgia of the ankle and/or foot', 'Arthralgia of the pelvic region and thigh', 'Arthralgia of the upper arm', 'Scalding pain on urination', 'Scapulalgia', 'Secondary physiologic amenorrhea', 'Seen by pediatrician', 'Seizure', 'Self-injurious behavior', 'Shadow of lung', 'Shoulder joint deformity', 'Shoulder joint pain', 'Shoulder joint unstable', 'Shoulder pain', 'Shoulder stiff', 'Artificial lens present', 'Single live birth', 'Sinus headache', 'Sinus tachycardia', 'Skin sensation disturbance', 'Slowing of urinary stream', 'Smells of urine', 'Smoker', 'Snoring', 'Soft tissue swelling', 'Spasm', 'Spasm of back muscles', 'Spastic paraparesis', 'Spasticity', 'Speech problem', 'Aspiration of food', 'Spontaneous rupture of membranes', 'Sputum - symptom', 'Sputum retention', 'Staring', 'Stented coronary artery', 'Stiff neck', 'Stopped smoking', 'Stress', 'Stridor', 'Asthenia', 'Ataxia', 'Subjective tinnitus', 'Suicidal thoughts', 'Suprapubic pain', 'Surgical follow-up', 'Attacks of weakness', 'Swallowing painful', 'Swelling', 'Swelling of eyelid', 'Swelling of finger', 'Swelling of hand', 'Swelling of limb', 'Swelling of scrotum', 'Swollen abdomen', 'Symbolic dysfunction', 'Systolic dysfunction', 'Systolic murmur', 'Atypical chest pain', 'Atypical facial pain', 'Tachypnea', 'Teenage pregnancy', 'Tenderness', 'Term infant', 'Thigh pain', 'Thyroid function tests abnormal', 'Tibial torsion', 'Tibiofibular joint pain', 'Tight chest', 'Tight foreskin', 'Tinnitus', 'Tired', 'Tobacco user', 'Toe swelling', 'Toe-walking gait', 'Tooth loss', 'Toothache', 'Total urinary incontinence', 'Transient global amnesia', 'Transplant follow-up', 'Tremor', 'Ultrasound scan abnormal', 'Umbilical discharge', 'Unable to balance', 'Unable to concentrate', 'Underweight', 'Unemployed', 'Unplanned pregnancy', 'Unprotected sexual intercourse', 'Unsatisfactory living conditions', 'Unstable knee', 'Unsteady when walking', 'Unwanted fertility', 'Up-to-date with immunizations', 'Upper abdominal pain', 'Ureteric colic', 'Urethral discharge', 'Urge incontinence of urine', 'Urgent desire for stool', 'Urgent desire to urinate', 'Urinary incontinence', 'Urinary symptoms', 'Urine cytology abnormal', 'Urine screening abnormal', 'Urogenital finding', 'Vaginal delivery', 'Vaginal discharge', 'Vaginal irritation', 'Vaginal pain', 'Vaginal show', 'Vasospasm', 'Venous stasis', 'Vertigo', 'Victim of abuse', 'Victim of physical assault', 'Victim of sexual aggression', 'Victim of terrorism', 'Visual field defect', 'Visual field scotoma', 'Vocal cord dysfunction', 'Vulval pain', 'Walking disability', 'Weakness of hand', 'Wheezing', 'Worried well', 'Wound hematoma', 'Wound pain', 'Wrinkled skin', 'Wrist joint pain', 'Barium enema abnormal', 'Bitemporal hemianopia', 'Bleeding', 'Bleeding from nose', 'Bleeding from vagina', 'Bleeding gums', 'Blood in urine', 'Clinical finding', 'Dizziness', 'Adult victim of abuse', 'Alteration in parenting', 'Weight decreased', 'Weight increased', 'Body mass index 40+ - severely obese', 'Drug-induced hyperpyrexia', 'Hematochezia', 'Magnetic resonance imaging of brain abnormal', 'Already on aspirin', 'RhD negative', 'Chronic low back pain', 'Acute low back pain', 'Low back pain', 'Mechanical low back pain', 'Tachycardia', 'Thoracic back pain', 'Chronic back pain', 'Lagophthalmos', 'Mental health problem', 'Anal pain', 'Paralytic lagophthalmos', 'Partial thromboplastin time increased', 'Fetal distress, in liveborn infant', 'Fetal intrauterine distress first noted during labor AND/OR delivery in liveborn infant', 'Sweating', 'Cervicovaginal cytology normal or benign', 'Cervicovaginal cytology: High grade squamous intraepithelial lesion or carcinoma', 'Cervicovaginal cytology: Low grade squamous intraepithelial lesion', 'Delivery normal', 'Advanced maternal age gravida', 'Biliary colic', 'Unconscious', 'Backache', 'Victim of physical abuse', 'Breastfeeding problem in the newborn', 'Carotid bruit', 'Deformity', 'Delirious', 'Flatulence symptom', 'Itching', 'Altered mental status', 'Motion sickness', 'Narrow angle', 'Nightmares', "Raynaud's phenomenon", 'Tardy ulnar nerve palsy', 'Bereavement due to life event', 'Essential tremor', 'Elevated erythrocyte sedimentation rate', 'Serum cholesterol borderline', 'Communicable disease contact', 'Edema of leg', 'Hepatitis B contact', 'Blind', 'Bowing of leg', 'Increased thyroid stimulating hormone level', 'Depression', 'Finding of nocturia', 'Bloodshot eye', 'Leg cramp', 'Swelling of arm', 'Fluid volume deficit', 'Benign familial tremor', 'Regular drinker', 'Dry eye', 'Joint pain in ankle and foot', 'Sensation of pressure in ear', 'Unstable ankle', 'Economic problem', 'Increased blood eosinophil number', 'Reflux', 'Weakness of limb', 'Open angle with borderline findings (disorder)', 'Inadequate housing', 'Sensation of burning of skin', 'Vaccination required', 'Breast screening declined', 'Subretinal neovascularization', 'Swollen legs', 'Claudication', 'Abdominal distension', 'Self-harm', 'Premenopausal menorrhagia', 'Acquired unequal leg length', 'Hearing aid worn', 'Fecal occult blood: positive', 'Heart irregular', 'Immunization refused', 'Constantly crying baby', 'Injection of surface of eye', 'Arthralgia of the lower leg', 'Papule', 'Globus hystericus', 'Globus sensation', 'Sacral dimples', 'Dysfunctional uterine bleeding', 'Special educational needs', 'Sexual assault', 'Oral contraceptive prescribed', 'Unsteady gait', 'Child relationship problem', 'Urolith', 'Deficient knowledge', 'Knee pain', 'Complaining of debility and malaise', 'Ill-defined disease', 'Intra-abdominal and pelvic swelling, mass and lump', 'Otalgia', 'Current non-smoker', 'Tonic-clonic seizure', 'Serum cholesterol borderline high', 'Dry skin', 'Glucose tolerance test during pregnancy - baby not yet delivered outside reference range', 'Post percutaneous transluminal coronary angioplasty', 'Threatened premature labor - not delivered', 'Habitual aborter - not delivered', 'Swallowing problem', 'Genitourinary symptoms', 'Lithium monitoring', 'General symptom', 'School problem', 'Standard chest X-ray abnormal', 'Perinatal jaundice due to hereditary hemolytic anemia', 'Perinatal jaundice due to galactosemia', 'Anxiety neurosis', 'Perinatal jaundice due to excessive hemolysis', 'Lytic lesion of bone on X-ray', 'Laboratory finding abnormal (navigational concept)', 'Maternal concern']
"""
