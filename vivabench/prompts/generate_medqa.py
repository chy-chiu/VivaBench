# TODO: Fix the pydantic dataclass descriptor + output examples here if we use MedQA again in large scale
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

MEDQA_IMAGING_PARSE_EXPAND_PROMPT = """You are a medical AI assistant. Given a short clinical vignette from a USMLE examination question, you are to parse the imaging mentioned in the vignette, and potentially expand the list of imaging done to fit the patient's clinical picture as you see fit. For each generated imaging modality, return a paragraph of radiological findings that would appear in a clinical note for this patient. Do not include the diagnosis in your findings / report text. Additionally, your goal is to minimize excessive investigations. If the patient's diagnosis does not require imaging to confirm, you should not augment any imaging modalities not mentioned in the input. However, if the patient was diagnosed with / presenting with findings that wouuld have radiological findings, you should include them. Do not include any additional history, examination findings, or other investigation findings.

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

MEDQA_INVESTIGATION_PARSE_PROMPT = """You are a medical AI assistant. Given a short clinical vignette with both original and augmented investigation results, you are to parse the investigation results to fit the patient's clinical picture. Reply in .json format, with data class format Investigations = {"bedside": Dict[str, InvestigationResult], "blood": Dict[str, InvestigationResult], "urine": Dict[str, InvestigationResult], ...}. Do not include any comments / calculations in your .json output. 

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
