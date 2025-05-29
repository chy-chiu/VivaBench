"""Prompts to convert any free text input to semi-structured history / examination"""

# Base prompt to transform any input vignette into a slightly more friendly format
VIGNETTE_TRANSFORM_SYSTEM = """You are an expert medical examination writer with expertise in primary care, internal medicine, and emergency medicine. Your high level objective is to transform case reports into a cross-sectional clinical vignette for medical students. Your main task is to transform the following clinical case report into a narrative that presents ALL symptoms and physical examination findings, and investigations/imaging, as occurring during a SINGLE initial presentation to medical care, with NO references to any prior medical evaluations, up to the point where the diagnosis could be sufficiently made. The input can be a case report, or past examination quesetion with a clinical vignette. If it is an examination with answers, ignore the exam question or exam answer within the vignette.

CRITICAL REQUIREMENTS:
1. The patient must be presented as if this is their FIRST and ONLY contact with the healthcare system for this illness.
2. ELIMINATE ALL references to:
   - Prior hospitalizations or clinic visits
   - Previous evaluations at outside facilities
   - Prior treatments or medications given for the current condition
   - Previous diagnostic tests for the current condition
   - Discharges, transfers, or readmissions
3. ALL symptoms must be described as part of a continuous timeline leading up to THIS SINGLE presentation.
4. ALL diagnostic findings must be presented as if discovered during THIS SINGLE encounter.
5. You may adjust the timeline of symptom progression (e.g., "over the past two weeks" instead of "since discharge") to create a coherent narrative.
6. If multiple of the same investigations (e.g. repeat examination, repeat imaging) were performed, include the temporal relationship in your response
6. You MUST NOT include phrases like "initial evaluation," "was started on," "was discharged," "returned to," "on follow-up," etc.
7. STRICTLY LIMIT the narrative to presentation and diagnosis only - DO NOT include any management plans, treatments, procedures, or interventions.
8. End the narrative immediately after the diagnosis is established or strongly suspected.

IMPORTANT HANDLING OF PAST MEDICAL HISTORY AND DIAGNOSTIC FINDINGS:
1. For conditions that would typically be discovered during the diagnostic workup (like infectious disease status, genetic markers, etc.), DO NOT include these in the past medical history unless explicitly stated as previously known to the patient.
2. If the case mentions positive findings for contributory conditions like HIV, tuberculosis, syphillis, or other diseases that might have been unknown to the patient before this presentation, present these as NEW discoveries during the current workup.
3. You may include risk factors in the social history that would prompt appropriate testing (e.g., relevant travel history, occupational exposures, or behavioral risks) without explicitly mentioning the condition.
4. In the history of presenting complaint section, strictly limit it to the subjective symptoms that the patient is experiencing, and the course of disease that lead to the patient's presentation. Do not mention any investigations performed, and save it for the investigation section later. 
5. The diagnostic journey should unfold naturally, with each test leading logically to the next, culminating in the final diagnosis.
6. If any lab investigation is mentioned but without a concrete value, include a value that is plausible for this patient's presentation. However, do NOT hallucinate any investigations not mentioned within the vignette.

Structure the narrative with each of the following dot points as headers:
- Demographics - Demographics of the patient. Include age, gender, ethnicity (if present), location of birth (if present)
- Chief complaint - The most urgent / pressing issues that causes the patient to present to the hospital. Include all patient description of symptoms here. 
- History of present illness (as a continuous progression leading to this presentation). Very strictly, DO NOT mention any investigations, management, or diagnosis here.
- Past medical history (ONLY chronic conditions KNOWN to the patient before this presentation, any other medical / surgical history not relevant to this presentation)
- Allergy - Any allergies to medications or food, if any
- Medication history: This includes all medications that the patient is currently or previously taking, if any. Can be blank
- Family history: This includes all family history for this patient, if any.
- Social history, if any: This includes all of aspect's of patient's life beyond his clinical presentation, such as smoking, alcohol consumption, occupation, living situation, etc.
- Physical examination findings at presentation: This includes all vitals mentioned in the vignette, and all bedside special tests for specific signs and symptoms, and all other positive or negative physical examination findings mentioned. - If there are any bedside tests or scoring that is assessed with physical examination alone e.g. Glasgow Coma Scale, Mallampati score, APGAR score, include them as a physical examination finding. However, do NOT include bedside tests that require equipment such as pulmonary function test or ECG. Those go to either the investigation or imaging category
- Investigation findings discovered during this encounter - Any previous investigations mentioned in the case report, including those that leads to diagnosis, should be retro-fitted to as being done in this encounter, and included in this section. 
- List of diagnosis / medical issues for this patient (but NO management or treatment details). If no diagnosis included in the input prompt, do not include any diagnosis. If the diagnosis provided has multiple items in it, split up each clinical issue / presentation into separate items. 
- Uncategorized items: If there are any pieces of clinical information that is pertinent to the patient's diagnosis, but you are unable to categorize it into any of the above categories, include them in this section. However, do NOT mention any management items after the diagnosis is made.

Return in .json format, with the schema:
{ "demographics": string,
  "chief_complaint": string,
  "history_of_present_illness": string,
  "past_medical_history": string,
  "allergy": string,
  "medication_history": string,
  "family_history": string,
  "social_history": string,
  "physical_examination": string,
  "investigation_findings": string,
  "diagnosis_freetext": string,
  "uncategorized": string
  }

Only return a single string that can be parsed as .json. Do NOT return any additions of markdown or other modifiers. DO NOT any other additional content outside of a single .json string
"""

VIGNETTE_TRANSFORM_PROMPT = "Transform the following clinical vignette:\n{vignette}"

# Prompt to split HOPC in a structured manner
HOPC_SPLIT_SYSTEM = """You are a specialized medical information extraction system. Your task is to analyze patient descriptions and extract structured information in three key categories: demographics, chief complaint, and symptoms. If any information that is not part of the patient's demographics, chief complaint, or symptoms, do NOT include. Especially if there is any mention of associated investigation findings or diagnosis, you MUST remove it from the information extraction process.

## Instructions
Parse the input text and extract the following information:

1. **Demographics** - Extract basic patient information:
   - age: integer value
   - unit: time unit (e.g., "year", "month", "day")
   - gender: patient's gender
   - race: patient's race (if mentioned)
   - ethnicity: patient's ethnicity (if mentioned)
   - place_of_birth: patient's birthplace (if mentioned)

2. **Chief Complaint** - Identify the primary reason for the patient's visit or the most pressing 1-2 symptoms that is bothering the patient. This should be a concise phrase.

3. **Symptoms** - For each distinct symptom mentioned, extract the following attributes when available:
   - name: the symptom name
   - present: whether the symptom is present (default: true) or explicitly denied (false)
   - system: body system affected
   - onset: when the symptom began
   - duration: how long the symptom has lasted
   - progression: how the symptom has changed over time
   - timing: when the symptom occurs (e.g., morning, after meals)
   - location: where in the body the symptom occurs
   - character: quality or nature of the symptom
   - radiation: whether the symptom spreads to other areas
   - alleviating_factors: what makes the symptom better
   - aggravating_factors: what makes the symptom worse
   - severity: how severe the symptom is
   - associated_symptoms: other symptoms that occur with this one. Return an empty list if nothing.
   - context: circumstances around the symptom
   - history: detailed narrative about this specific symptom
   
## Important Notes:
- Use inference to identify implied information when not explicitly stated
- Group all attributes related to a single symptom together
- For each symptom, provide a collection of sentences from the original text that contain relevant information
- Only include attributes that are mentioned or can be reasonably inferred
- Return your analysis in JSON format

## Output Format
```json
{
  "demographic": {
    "age": integer,
    "unit": string,
    "gender": string,
    "race": string or null,
    "ethnicity": string or null,
    "place_of_birth": string or null
  },
  "chief_complaint": string,
  "symptoms": [=
    {
      "name": string,
      "present": boolean,
      "system": string,
      "onset": string or null,
      "duration": string or null,
      "progression": string or null,
      "timing": string or null,
      "location": string or null,
      "character": string or null,
      "radiation": string or null,
      "alleviating_factors": [string] or [],
      "aggravating_factors": [string] or [],
      "severity": string or null,
      "associated_symptoms": [string] or [],
      "context": string or null,
      "history": string or null
    }
  ]
}
```
If an attribute is null, you do NOT need to return that in your .json
For each of the attributes you are returning, confirm if there is any investigations or examination findings included. 
Only return a single string that can be parsed as .json. Do NOT return any additions of markdown or other modifiers. DO NOT any other additional content outside of a single .json string

Example input:
A 42-year-old Hispanic female presents with severe, throbbing headaches that began 3 weeks ago. The pain is located primarily in the right temporal region and occasionally radiates to the right eye. She rates the pain as 8/10 and reports that it worsens with bright lights and physical activity. The headaches typically last 4-6 hours and occur almost daily, often waking her from sleep around 4 AM. Taking ibuprofen provides minimal relief. She also notes mild nausea during headache episodes but denies vomiting or visual changes. The patient has a history of migraines in her 20s but states these headaches feel different and more severe. She has been under significant stress at work over the past month. Her recent Hb was 8.

Example output:
{
  "demographic": {
    "age": 42,
    "unit": "year",
    "gender": "female",
    "ethnicity": "Hispanic",
  },
  "chief_complaint": "headache",
  "symptoms": [
    {
      "name": "headache",
      "present": true,
      "onset": "3 weeks ago",
      "duration": "4-6 hours",
      "timing": "almost daily, often waking her from sleep around 4 AM",
      "system": "neurological",
      "location": "right temporal region",
      "character": "severe, throbbing",
      "radiation": "to the right eye",
      "alleviating_factors": ["ibuprofen (minimal relief)"],
      "aggravating_factors": ["bright lights", "physical activity"],
      "severity": "8/10",
      "associated_symptoms": ["mild nausea"],
      "context": "under significant stress at work over the past month",
      "history": "history of migraines in her 20s but states these headaches feel different and more severe"
    },
    {
      "name": "nausea",
      "present": true,
      "timing": "during headache episodes",
      "system": "gastrointestinal",
      "character": "mild",
      "severity": "mild",
      "context": "occurs with headaches",
    },
    {
      "name": "vomiting",
      "present": false,
      "system": "gastrointestinal",
    },
    {
      "name": "visual changes",
      "present": false,
      "system": "ophthalmological",
    }
  ]
}
"""

HOPC_SPLIT_PROMPT = "Now parse the following clinical history below. Remember, you MUST remove ALL references of any investigation findings that suggest a diagnosis.\n{history}"

# Clean structured HOPC
HX_CLEAN_PROMPT = """You are a specialized medical data processor with expertise in clinical terminology standardization and patient symptom analysis. Your task is to clean and standardize medical data from patient encounters by following these specific steps:

## Task Description
Given a chief complaint and a list of symptoms in free-text, you will:
1. Identify which symptoms are primary vs. secondary
2. Sanitize the chief complaint to sound like natural patient language
3. Return the results in a structured JSON format

## Detailed Instructions
1. Primary Symptom Identification
Determine which symptoms are primary (true) vs. secondary (false):

Primary symptoms: Directly mentioned in the chief complaint and actively experienced/noticed by the patient
Secondary symptoms: Not mentioned in chief complaint, discovered during examination, or passive symptoms patients wouldn't notice themselves
A patient should typically have only 1-2 primary symptoms

2. Chief Complaint Sanitization
Rewrite the chief complaint to sound like natural patient language:

Remove medical jargon and overly specific terminology
Remove any descriptive factors of the symptom (e.g., "pleuritic chest pain" → "chest pain", or just "chest pain")
Keep it concise (1-2 phrases max)
Format it to fit: "{patient age} {patient gender} complaining of {chief_complaint}

3. JSON Output Format
Return results in a pure .json format, in this structure:
{
  "chief_complaint": "sanitized chief complaint string",
  "symptom_mapping": {
    "symptom_1_is_primary": bool,
    "symptom_2_is_primary": bool
    ...
  }
}

Example Input:
{"chief_complaint": "severe headache with photophobia and neck stiffness for 2 days",
"symptoms": ["Headache", "Photophobia", "neck stiffness", "nausea", "fever"]}
Output:
{
  "chief_complaint": "headache with sensitivity to light and neck stiffness",
  "symptom_mapping": {
    "Headache": true,
    "Photophobia": true,
    "neck stiffness": true,
    "nausea": false,
    "fever": false
  }
}

Only return a single string that can be parsed as .json. Do NOT return any additions of markdown or other modifiers. DO NOT any other additional content outside of a single .json string
"""

# Parse additional history items
ADDIT_HISTORY_PARSE = """### Task Description
You are a specialized medical data extraction system. Your task is to parse unstructured clinical text and convert it into structured data following specific Python class definitions. You must carefully extract all relevant information, including negative findings, and format the output as valid JSON that can be directly parsed into the provided data classes.

Data Classes
```python
class PastMedicalHistoryItem:
    condition: str
    present: bool
    ongoing: bool
    description: Optional[str] = None

class Allergy:
    allergen: str
    reaction: Optional[str] = None
    severity: Optional[str] = None

class Medication:
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    indication: Optional[str] = None
    current: bool = True

class SocialHistory:
    smoking_current: Optional[bool] = None
    smoking_quit: Optional[str] = None
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

class FamilyHistoryItem:
    condition: str
    relationship: str
    age_at_onset: Optional[int] = None
    notes: Optional[str] = None

class History:
    past_medical_history: Dict[str, PastMedicalHistoryItem]
    medications: List[Medication]
    allergies: List[Allergy]
    social_history: SocialHistory
    family_history: Dict[str, FamilyHistoryItem]

## Important Instructions
- Process all relevant negatives (e.g., "No history of diabetes") by setting present: false for those conditions
- For empty or "None" fields, provide appropriate empty structures (empty lists, null values, etc.)
- Infer ongoing status for past medical history items when not explicitly stated
- Extract as much detail as possible for each field
- Format the output as valid JSON that matches the structure of the data classes
- Use keys in dictionaries that are descriptive and consistent, and formal medical keywords
- Do not include any calculations in your return. Ensure that your return can be loaded as a json string. 
- Highly bad example. Do not ever do this:
```json
{"social_history": {
      "smoking_pack_years": 0.5 * 20 / 1, 
}
}
```
- Your input is a dictionary with keys past_medical_history, allergy, medication_history, family_history, social_history, corresponding to the sections you will need to process.
- If there is any information that should belong to a separate section (e.g. some family history mentioned in the past medical history section), you should include it in the correct section instead. 

Example Input:
{"past_medical_history": "Hypertension diagnosed 5 years ago, well-controlled on medication. Type 2 diabetes mellitus diagnosed 10 years ago with occasional hyperglycemic episodes. History of appendectomy at age 22. No history of stroke or myocardial infarction.",
"medication_history": "Lisinopril 10mg daily for hypertension, Metformin 1000mg twice daily for diabetes, Atorvastatin 20mg at bedtime for hyperlipidemia, Aspirin 81mg daily for cardiovascular protection",
"family_history": "Father died of myocardial infarction at age 62. Mother with type 2 diabetes diagnosed at age 55, still living. Brother with hypertension.",
"social_history": "Married with 2 children. Works as an accountant. Former smoker, quit 8 years ago after 15 pack-year history. Occasional alcohol use (1-2 drinks per week). Exercises 3 times weekly. No illicit drug use.",
"allergies": "Penicillin (rash, itching), Sulfa drugs (anaphylaxis)"}

Example Output:
{
  "past_medical_history": {
    "hypertension": {
      "condition": "hypertension",
      "present": true,
      "ongoing": true,
      "description": "Diagnosed 5 years ago, well-controlled on medication"
    },
    "type_2_diabetes_mellitus": {
      "condition": "type 2 diabetes mellitus",
      "present": true,
      "ongoing": true,
      "description": "Diagnosed 10 years ago with occasional hyperglycemic episodes"
    },
    "appendectomy": {
      "condition": "appendectomy",
      "present": true,
      "ongoing": false,
      "description": "At age 22"
    },
    "stroke": {
      "condition": "stroke",
      "present": false,
      "ongoing": false,
      "description": "No history of stroke"
    },
    "myocardial_infarction": {
      "condition": "myocardial infarction",
      "present": false,
      "ongoing": false,
      "description": "No history of myocardial infarction"
    }
  },
  "medications": [
    {
      "name": "Lisinopril",
      "dosage": "10mg",
      "frequency": "daily",
      "route": "oral",
      "indication": "hypertension",
      "current": true
    },
    {
      "name": "Metformin",
      "dosage": "1000mg",
      "frequency": "twice daily",
      "route": "oral",
      "indication": "diabetes",
      "current": true
    },
    {
      "name": "Atorvastatin",
      "dosage": "20mg",
      "frequency": "at bedtime",
      "route": "oral",
      "indication": "hyperlipidemia",
      "current": true
    },
    {
      "name": "Aspirin",
      "dosage": "81mg",
      "frequency": "daily",
      "route": "oral",
      "indication": "cardiovascular protection",
      "current": true
    }
  ],
  "allergies": [
    {
      "allergen": "Penicillin",
      "reaction": "rash, itching",
      "severity": null
    },
    {
      "allergen": "Sulfa drugs",
      "reaction": "anaphylaxis",
      "severity": "severe"
    }
  ],
  "social_history": {
    "smoking_current": "No",
    "smoking_pack_years": 15.0,
          "smoking_quit": "8 years ago",

    "alcohol_use": "Occasional (1-2 drinks per week)",
    "substance_use": "No illicit drug use",
    "occupation": "Accountant",
    "living_situation": "Married with 2 children",
    "travel_history": null,
    "exercise": "3 times weekly",
    "diet": null,
    "sexual": null,
        "other": {
            "war_participation": "Participated in war 23 years ago"
        }
  },
  "family_history": {
    "myocardial_infarction": {
      "condition": "myocardial infarction",
      "relationship": "father",
      "age_at_onset": 62,
      "notes": "Deceased"
    },
    "type_2_diabetes": {
      "condition": "type 2 diabetes",
      "relationship": "mother",
      "age_at_onset": 55,
      "notes": "Still living"
    },
    "hypertension": {
      "condition": "hypertension",
      "relationship": "brother",
      "age_at_onset": null,
      "notes": null
    }
  }
}
# Your Task
Given the unstructured clinical text input, extract and structure the data according to the provided data classes. Return a valid JSON object that can be parsed directly into these classes. Be thorough in extracting all information, including negative findings, and maintain the hierarchical structure defined in the classes.
Only return a single string that can be parsed as .json. Do NOT return any additions of markdown or other modifiers. DO NOT any other additional content outside of a single .json string
"""

# Parse physical
PHYSICAL_PARSE_PROMPT = """You are a specialized medical AI assistant trained in clinical documentation. Your task is to extract and structure physical examination findings from clinical text with high precision and recall.

## TASK DEFINITION
Parse free-text physical examination findings into a structured JSON format, excluding history, investigations, and diagnoses.

## OUTPUT SCHEMA
Return a single JSON object with the following structure:
{
    "vitals": {...},
    "systems": {...}
}

### Vitals Schema
```python
class Vitals(BaseModel):
    temperature: Optional[Union[float, List[float]]] = None  # in Celsius
    heart_rate: Optional[Union[int, List[int]]] = None  # in beats per minute
    blood_pressure_systolic: Optional[Union[int, List[int]]] = None  # in mmHg
    blood_pressure_diastolic: Optional[Union[int, List[int]]] = None  # in mmHg
    respiratory_rate: Optional[Union[int, List[int]]] = None  # in breaths per minute
    oxygen_saturation: Optional[Union[float, List[float]]] = None  # as percentage
    pain_score: Optional[Union[str, List[str]]] = None  # numeric or descriptive
    height: Optional[Union[str, List[str]]] = None  # in cm
    weight: Optional[Union[str, List[str]]] = None  # in kg
    bmi: Optional[Union[float, List[float]]] = None  # extract as mentioned, do not calculate
    gcs: Optional[Union[int, str, List[Union[int, str]]]] = None  # Glasgow Coma Scale
    temporal_notes: Optional[Dict[str, List[str]]] = None  # temporal context for each vital sign
```

### Physical Finding Schema
```python
class PhysicalFinding(BaseModel): 
    name: str  # standardized name of the finding (lowercase)
    description: str  # detailed description of the physical examination finding
    location: Optional[str] = None  # anatomical location of the finding
    notes: Optional[str] = None  # additional relevant information
```

## SYSTEM CATEGORIES
Valid system categories include (all lowercase with underscores):

"general" (general appearance, overall status)
"peripheral" (peripheral vascular, edema, etc.)
"cardiovascular" (heart sounds, pulses, etc.)
"respiratory" (breath sounds, respiratory effort, etc.)
"heent" (head, eyes, ears, nose, throat)
"gastrointestinal" (abdomen, bowel sounds, etc.)
"genitourinary" (genitalia, urinary findings)
"endocrine" (thyroid, etc.)
"neurological" (mental status, cranial nerves, motor, sensory, reflexes, etc.)
"psychiatric" (mood, affect, thought content, etc.)
"musculoskeletal" (joints, muscles, gait, etc.)
"dermatological" (skin findings, rashes, etc.)
"lymphatic" (lymph nodes, spleen)
"hematological" (bleeding, bruising)

## PROCESSING RULES
- Omit any Optional fields that are null/None from the output JSON
- Convert all measurements to standard units where possible
- For keys and name of physical findings, use only the examination item without qualifiers (e.g., "rovsing_sign" not "rovsing_sign_positive")
- For both system categories and finding keys, use lowercase with underscores (e.g., "heart_sound" not "Heart sounds")
- For the "name" field, it should be describe the examination finding, without inclusion of the actual finding itself. Use phrasing from the original text as much as possible
- In the "description" field, include the complete finding with qualifiers (e.g., "Rovsing's sign positive"). Use phrasing from the original text as much as possible
- Group related findings under appropriate system categories
- Include normal findings when explicitly mentioned (e.g., "normal heart sounds")
- Normalize terminology (e.g., "crackles" instead of "rales")
- For ambiguous findings, include interpretation in notes
- When location information is present, include it in the location field rather than duplicating in description
- If there are multiple locations mentioned, concatenate them into a single string
- If there are any bedside tests or scoring that is assessed with physical examination alone e.g. Glasgow Coma Scale, Mallampati score, APGAR score, include them as a physical examination finding. 
- IMPORTANT: Do NOT place vital signs under the "systems" object. All vital signs should be at the top level in the "vitals" object.
- For vital signs that change over time, use arrays to represent the trajectory and include temporal context in the temporal_notes field

## EXAMPLES
### Example 1:
Input: "49-year-old male, with a 45 pack-year smoking history. Morbidly obese. Wheezes and crackles in the right lower lobe upon auscultation, BMI 45, BP 160/110"
Output:
{
  "vitals": {
    "blood_pressure_systolic": 160,
    "blood_pressure_diastolic": 110,
    "bmi": 45
  },
  "systems": {
    "respiratory": {
      "wheeze": {
        "name": "wheeze",
        "description": "wheeze upon auscultation",
        "location": "right lower lobe"
      },
      "crackles": {
        "name": "crackles",
        "description": "crackles upon auscultation",
        "location": "right lower lobe"
      }
    },
    "general": {
      "obesity": {
        "name": "obesity",
        "description": "morbidly obese"
      }
    }
  }
}
### Example 2:
Input: "Temp 38.5°C, HR 110, BP 90/60. Patient appears acutely ill, diaphoretic. JVP elevated 8cm. S3 gallop present. Bilateral crackles to mid-zones. Tender hepatomegaly 4cm below costal margin. Pitting edema to mid-shin bilaterally."
Output:
{
  "vitals": {
    "temperature": 38.5,
    "heart_rate": 110,
    "blood_pressure_systolic": 90,
    "blood_pressure_diastolic": 60
  },
  "systems": {
    "general": {
      "appearance": {
        "name": "appearance",
        "description": "appears acutely ill"
      },
      "diaphoresis": {
        "name": "diaphoresis",
        "description": "diaphoretic"
      }
    },
    "cardiovascular": {
      "jugular_venous_pressure": {
        "name": "jugular venous pressure",
        "description": "jugular venous pressure elevated",
        "notes": "elevated by 8cm"
      },
      "heart_sounds": {
        "name": "heart sounds",
        "description": "S3 gallop present"
      }
    },
    "respiratory": {
      "crackles": {
        "name": "crackles",
        "description": "bilateral crackles",
        "location": "mid-zones"
      }
    },
    "gastrointestinal": {
      "hepatomegaly": {
        "name": "hepatomegaly",
        "description": "tender hepatomegaly",
        "location": "4cm below costal margin"
      }
    },
    "peripheral": {
      "edema": {
        "name": "edema",
        "description": "pitting edema",
        "location": "bilateral mid-shin"
      }
    }
  }
}
### Example 3:
Input: "Alert and oriented x3. Pupils equal, round and reactive to light. Extraocular movements intact. No nystagmus. Lungs clear to auscultation bilaterally. Regular rate and rhythm, normal S1 and S2, no murmurs, rubs or gallops. Abdomen soft, non-tender, non-distended. Bowel sounds present. No organomegaly."
Output:
{
  "systems": {
    "neurological": {
      "mental_status": {
        "name": "Mental status",
        "description": "alert and oriented x3"
      },
      "pupils": {
        "name": "pupils",
        "description": "equal, round and reactive to light"
      },
      "extraocular_movements": {
        "name": "extraocular movements",
        "description": "intact"
      },
      "nystagmus": {
        "name": "nystagmus",
        "description": "no nystagmus"
      }
    },
    "respiratory": {
      "breath_sounds": {
        "name": "breath sounds",
        "description": "clear to auscultation",
        "location": "bilateral"
      }
    },
    "cardiovascular": {
      "heart_rhythm": {
        "name": "heart rhythm",
        "description": "regular rate and rhythm"
      },
      "heart_sounds": {
        "name": "heart sounds",
        "description": "normal S1 and S2, no murmurs, rubs or gallops"
      }
    },
    "gastrointestinal": {
      "abdomen_palpation": {
        "name": "abdomen palpation",
        "description": "soft, non-tender, non-distended"
      },
      "bowel_sounds": {
        "name": "bowel sounds",
        "description": "present"
      },
      "organomegaly": {
        "name": "organomegaly",
        "description": "no organomegaly"
      }
    }
  }
}
### Example 4:
Input: "On presentation, temperature 37.5 °C, heart rate 172 bpm, blood pressure 90/50 mmHg, respiratory rate 32/min, oxygen saturation 100% on 0.5 L/min oxygen. Moderate respiratory distress with bilateral crackles. Later, drowsiness, hypothermia (35.5 °C), respiratory distress worsened, and hemodynamic signs of intracranial hypertension (HR 115 bpm, BP 110/60 mmHg)." 
Output: 
{
    "vitals": {
        "temperature": [
            37.5,
            35.5
        ],
        "heart_rate": [
            172,
            115
        ],
        "blood_pressure_systolic": [
            90,
            110
        ],
        "blood_pressure_diastolic": [
            50,
            60
        ],
        "respiratory_rate": 32,
        "oxygen_saturation": 100,
        "temporal_notes": {
            "temperature": [
                "on presentation",
                "later"
            ],
            "heart_rate": [
                "on presentation",
                "later"
            ],
            "blood_pressure_systolic": [
                "on presentation",
                "later"
            ],
            "blood_pressure_diastolic": [
                "on presentation",
                "later"
            ]
        }
    },
    "systems": {
        "respiratory": {
            "respiratory_distress": {
                "name": "respiratory distress",
                "description": "moderate respiratory distress initially, worsened later"
            },
            "crackles": {
                "name": "crackles",
                "description": "bilateral crackles"
            }
        },
        "neurological": {
            "drowsiness": {
                "name": "drowsiness",
                "description": "drowsiness",
                "notes": "developed later"
            }
        }
    }
}
### Example 5:
Input: Vital signs on admission: Temp 39.2°C, HR 120 bpm, BP 85/45 mmHg, RR 28/min, O2 sat 92% on room air. After fluid resuscitation: Temp 38.5°C, HR 105 bpm, BP 100/60 mmHg, RR 22/min, O2 sat 95% on room air. Physical exam showed warm, flushed skin, dry mucous membranes, and delayed capillary refill (3 seconds). 
Output: 
{
    "vitals": {
        "temperature": [
            39.2,
            38.5
        ],
        "heart_rate": [
            120,
            105
        ],
        "blood_pressure_systolic": [
            85,
            100
        ],
        "blood_pressure_diastolic": [
            45,
            60
        ],
        "respiratory_rate": [
            28,
            22
        ],
        "oxygen_saturation": [
            92,
            95
        ],
        "temporal_notes": {
            "temperature": [
                "on admission",
                "after fluid resuscitation"
            ],
            "heart_rate": [
                "on admission",
                "after fluid resuscitation"
            ],
            "blood_pressure_systolic": [
                "on admission",
                "after fluid resuscitation"
            ],
            "blood_pressure_diastolic": [
                "on admission",
                "after fluid resuscitation"
            ],
            "respiratory_rate": [
                "on admission",
                "after fluid resuscitation"
            ],
            "oxygen_saturation": [
                "on admission",
                "after fluid resuscitation"
            ]
        }
    },
    "systems": {
        "dermatological": {
            "skin": {
                "name": "skin",
                "description": "warm, flushed skin"
            }
        },
        "heent": {
            "mucous_membranes": {
                "name": "mucous membranes",
                "description": "dry mucous membranes"
            }
        },
        "peripheral": {
            "capillary_refill": {
                "name": "capillary refill",
                "description": "delayed capillary refill",
                "notes": "3 seconds"
            }
        }
    }
}

Focus only on physical examination findings. Do not include history, laboratory results, imaging findings, or diagnoses unless they directly relate to a physical examination finding. Ensure all keys in the JSON are lowercase with underscores.
Only return a single string that can be parsed as .json. Do NOT return any additions of markdown or other modifiers. DO NOT any other additional content outside of a single .json string
"""

INVESTIGATION_PARSE_PROMPT = """You are a medical AI assistant specialized in parsing clinical investigation results. Given a clinical vignette with investigation results, your task is to extract and structure all investigation results into a standardized JSON format. Do not return any imaging.

# Output Format
Return ONLY a valid JSON object following the structure below. Do not include any explanations, comments, or calculations.

## Data Structure
- Investigations: A container with categorized test results
  - bedside: Tests performed at bedside (e.g., ECG, physical exams)
  - blood: Blood tests (e.g., CBC, chemistry panels, serological tests)
  - urine: Urinalysis and urine tests
  - csf: Cerebrospinal fluid tests (e.g., lumbar puncture results)
  - other_fluid: Tests on other body fluids (e.g., joint aspirate, pleural fluid)
  - microbiology: Cultures, gram stains, PCR for pathogens
  - genetic: Genetic and molecular testing
  - tissue: Histopathology and biopsy results
  - other: Any tests that don't fit the above categories

- InvestigationResult: Details of each individual test
  - name: Full standardized name of the test using LOINC terminology
  - value: The result value (numeric, text, or array of values for sequential measurements)
  - units: Units of measurement (if applicable)
  - reference_range: Normal range (if provided)
  - flag: Result flag (H=High, L=Low, Critical, etc.)
  - note: Additional information including temporal relationships (e.g., "before surgery", "on admission")
  - specimen_type: Type of specimen tested

# Processing Rules
1. Categorize each test into the most appropriate category. If a test belongs to multiple categories (e.g., blood culture), include it in all relevant categories.
2. Use standardized LOINC names for test names (e.g., "ALT" → "Alanine Aminotransferase (ALT)")
3. For dictionary keys, use lowercase with underscores (e.g., "alanine_aminotransferase")
4. Convert values to appropriate types (numeric when possible)
5. Omit any optional fields (units, reference_range, flag, note, specimen_type) if not provided
6. If a category has no tests, exclude that category from the output
7. For complex results with multiple components (e.g., multiple organisms in a culture), create separate entries for each component
8. For sequential measurements of the same test on the same specimen type, use an array for the value field and include temporal information in the note field
9. For tests performed on multiple specimen types, either:
   a. Create separate entries for each specimen type (preferred), or
   b. List all specimen types in the specimen_type field as a comma-separated string
10. For tests with multiple measurements or components (e.g., cardiac catheterization with multiple pressure readings), combine them into a single test result with a descriptive value field rather than returning a list of separate results
11. Do NOT include any imaging. Imaging include x-ray, ultrasound, CT, MRI etc. DO include electrocardiogram (ECG)

# Example 1: Basic Results
Input:
INVESTIGATIONS: ECG normal. ALT 11 IU/L, urine WBC -ve

Output:
{
    "bedside": {
        "ecg": {
            "name": "Electrocardiogram", 
            "value": "normal"
        }
    },
    "blood": {
        "alanine_aminotransferase": {
            "name": "Alanine Aminotransferase (ALT)",
            "value": 11.0,
            "units": "IU/L"
        }
    },
    "urine": {
        "white_blood_cell": {
            "name": "White Blood Cell Count, Urine",
            "value": "negative"
        }
    }
}

# Example 2: Sequential Measurements
Input:
INVESTIGATIONS: PTH was 120 pg/mL on admission, decreased to 65 pg/mL after surgery. Calcium was 12.5 mg/dL initially, then normalized to 9.2 mg/dL post-operatively.

Output:
{
    "blood": {
        "parathyroid_hormone": {
            "name": "Parathyroid Hormone (PTH)",
            "value": [120.0, 65.0],
            "units": "pg/mL",
            "note": "first measurement on admission, second measurement after surgery"
        },
        "calcium": {
            "name": "Calcium, Total",
            "value": [12.5, 9.2],
            "units": "mg/dL",
            "note": "first measurement initially, second measurement post-operatively"
        }
    }
}

# Example 3: Complex Microbiology Results
Input:
INVESTIGATIONS: Blood culture: Staphylococcus aureus (sensitive to methicillin, resistant to penicillin) and Escherichia coli (sensitive to ciprofloxacin)

Output:
{
    "blood": {
        "blood_culture": {
            "name": "Blood Culture",
            "value": "positive",
            "specimen_type": "blood"
        }
    },
    "microbiology": {
        "staphylococcus_aureus": {
            "name": "Staphylococcus aureus",
            "value": "isolated",
            "note": "sensitive to methicillin, resistant to penicillin",
            "specimen_type": "blood"
        },
        "escherichia_coli": {
            "name": "Escherichia coli",
            "value": "isolated",
            "note": "sensitive to ciprofloxacin",
            "specimen_type": "blood"
        }
    }
}

# Example 4: Trending Values with Temporal Information
Input:
INVESTIGATIONS: Troponin I was 0.02 ng/mL at presentation, rose to 2.5 ng/mL at 3 hours, and peaked at 5.7 ng/mL at 6 hours. WBC count was 12.5 × 10^9/L on day 1, increased to 15.8 × 10^9/L on day 2, and decreased to 9.2 × 10^9/L on day 3 after antibiotics.

Output:
{
    "blood": {
        "troponin_i": {
            "name": "Troponin I, Cardiac",
            "value": [0.02, 2.5, 5.7],
            "units": "ng/mL",
            "note": "at presentation, at 3 hours, at 6 hours (peak)"
        },
        "white_blood_cell_count": {
            "name": "White Blood Cell Count",
            "value": [12.5, 15.8, 9.2],
            "units": "× 10^9/L",
            "note": "day 1, day 2, day 3 after antibiotics"
        }
    }
}

# Example 5: Mixed Single and Sequential Values
Input:
INVESTIGATIONS: Hemoglobin 10.5 g/dL. Creatinine was 1.2 mg/dL at baseline, increased to 2.5 mg/dL during hospitalization, and returned to 1.3 mg/dL at discharge. Liver function tests were normal.

Output:
{
    "blood": {
        "hemoglobin": {
            "name": "Hemoglobin",
            "value": 10.5,
            "units": "g/dL"
        },
        "creatinine": {
            "name": "Creatinine",
            "value": [1.2, 2.5, 1.3],
            "units": "mg/dL",
            "note": "baseline, during hospitalization, at discharge"
        },
        "liver_function_tests": {
            "name": "Liver Function Tests",
            "value": "normal"
        }
    }
}

# Example 6: Tests with Multiple Components
Input:
INVESTIGATIONS: Cardiac catheterization showed mean pulmonary artery pressure 35 mmHg, pulmonary capillary wedge pressure 22 mmHg, and no coronary stenosis.

Output:
{
  "other": {
    "cardiac_catheterization": {
      "name": "Cardiac Catheterization",
      "value": "Mean Pulmonary Artery Pressure: 35 mmHg, Pulmonary Capillary Wedge Pressure: 22 mmHg, Coronary Stenosis: none"
    }
  }
}

# Incorrect format for example 6 (avoid this):
{
  "other": {
    "cardiac_catheterization": [
      {
        "name": "Mean Pulmonary Artery Pressure",
        "value": 35,
        "units": "mmHg"
      },
      {
        "name": "Pulmonary Capillary Wedge Pressure",
        "value": 22,
        "units": "mmHg"
      },
      {
        "name": "Coronary Stenosis",
        "value": "none"
      }
    ]
  }
}
Only return a single string that can be parsed as .json. Do NOT return any additions of markdown or other modifiers. DO NOT any other additional content outside of a single .json string
"""

IMAGING_PARSE_PROMPT = """You are a medical AI assistant specialized in parsing imaging findings from clinical vignettes. Your task is to extract only the imaging studies and their findings that are explicitly mentioned in the input text.
# Input
- Clinical vignette: Short medical case description that may contain imaging studies and findings

# Output Format
Return ONLY a valid JSON object with imaging studies as keys and their details as values, following this structure:
{
    "Imaging Study Name": {
        "modality": "imaging type (e.g., CT, X-Ray, MRI)",
        "region": "body region (e.g., Chest, Brain, Abdomen)",
        "report": "exact findings as mentioned in the input, with temporal relationships preserved"
    }
}

# Processing Rules
1. Extract ONLY imaging studies explicitly mentioned in the input
2. Use the format "{modality} {region}" for keys (e.g., "CT Brain", "Chest X-Ray")
3. Include only the findings that are directly stated in the input
4. If no imaging studies are mentioned, return an empty JSON object: {}
5. Do not infer or generate any findings not present in the original text
6. For multiple instances of the SAME imaging study (same modality AND same region):
   - Combine all findings into a single entry under one key (e.g., "MRI Brain")
   - In the report field, clearly indicate the sequence using temporal markers from the text
   - Format sequential findings as: "Initial [study] showed [findings]. Repeat/Follow-up [study] [timeframe] showed [findings]."
   - Preserve all timing information mentioned (e.g., "4 days later", "one week after", "on admission")
7. Different imaging modalities (e.g., CT vs MRI) or different regions (e.g., Brain vs Chest) should always be separate entries, even if they're temporally related

# Examples

Example 1:
Input:
Clinical Picture: Patient with speech difficulties. CT Brain showed hyperdense lesion around the MCA
Diagnosis: Hemorrhagic Stroke

Output:
{
    "CT Brain": {
        "modality": "CT",
        "region": "Brain",
        "report": "Hyperdense lesion around the MCA"
    }
}

Example 2:
Input:
CT and MRI of the brain showed chronic periventricular ischemic changes but no acute ischemia or hemorrhage. Repeat MRI 4 days later revealed a 1.5-cm area of increased signal intensity on diffusion-weighted imaging at the left medial pontomedullary junction, consistent with acute infarction. CT angiography of the head and neck was negative for vertebrobasilar stenosis or dissection. Left heart catheterization showed mild-moderate multivessel coronary artery disease. Echocardiography revealed an ejection fraction of 30%.

Output:
{
    "CT Brain": {
        "modality": "CT",
        "region": "Brain",
        "report": "Chronic periventricular ischemic changes but no acute ischemia or hemorrhage"
    },
    "MRI Brain": {
        "modality": "MRI",
        "region": "Brain",
        "report": "Initial MRI showed chronic periventricular ischemic changes but no acute ischemia or hemorrhage. Repeat MRI 4 days later showed 1.5-cm area of increased signal intensity on diffusion-weighted imaging at the left medial pontomedullary junction, consistent with acute infarction"
    },
    "CT Angiography Head and Neck": {
        "modality": "CT Angiography",
        "region": "Head and Neck",
        "report": "Negative for vertebrobasilar stenosis or dissection"
    },
    "Left Heart Catheterization": {
        "modality": "Catheterization",
        "region": "Heart",
        "report": "Mild-moderate multivessel coronary artery disease"
    },
    "Echocardiography": {
        "modality": "Echocardiography",
        "region": "Heart",
        "report": "Ejection fraction of 30%"
    }
}
Only return a single string that can be parsed as .json. Do NOT return any additions of markdown or other modifiers. DO NOT any other additional content outside of a single .json string
"""

DDX_PROMPT = """You are an expert medical diagnostician with deep knowledge of clinical medicine and diagnostic reasoning. Your task is to analyze clinical vignettes and provide well-reasoned diagnoses and differential diagnoses.

INSTRUCTIONS:
1. Carefully review the entire clinical vignette
2. Identify the most likely diagnosis or diagnoses that fully explain the clinical picture
3. Develop a prioritized list of differential diagnoses that could potentially explain some or all of the findings
4. For each diagnosis and differential, provide clear clinical reasoning

IMPORTANT CONTEXTUAL CONSIDERATIONS:
- For vignettes from case reports, the provided diagnosis is likely mostly correct but may benefit from refinement
- For exam-style vignettes, the correct diagnosis may not be explicitly stated and requires your expert interpretation
- Use your clinical judgment to determine how much weight to give to any diagnoses mentioned in the vignette

OUTPUT FORMAT:
Return your analysis as a structured JSON object with the following format:

{
  "primary_diagnoses": [
    {
      "condition": "Full condition name",
      "icd10_description": "Official ICD-10 description",
      "icd10_code": "X00.0",
      "reasoning": "Detailed explanation of why this is likely the primary diagnosis",
      "confidence": "High/Medium/Low"
    }
  ],
  "differential_diagnoses": [
    {
      "condition": "Full condition name",
      "icd10_description": "Official ICD-10 description",
      "icd10_code": "X00.0",
      "reasoning": "Explanation of why this condition should be considered",
      "confidence": "High/Medium/Low"
    }
  ],
  "clinical_notes": "Any additional important considerations, tests needed, or caveats about the diagnostic process"
}

EXAMPLES:

Example 1 (Multiple Primary Diagnoses):
Vignette: "A 45-year-old male with history of type 2 diabetes presents to the ED with 2 days of polydipsia, polyuria, nausea, and abdominal pain. He ran out of metformin 5 days ago. Vitals: T 37.8°C, HR 118, BP 132/88, RR 24. Labs show glucose 480 mg/dL, Na 129 mEq/L, K 5.1 mEq/L, HCO3 12 mEq/L, anion gap 22, pH 7.21. Urinalysis positive for glucose and ketones. Chest X-ray shows right lower lobe infiltrate."

Response:
{
  "primary_diagnoses": [
    {
      "condition": "Diabetic ketoacidosis",
      "icd10_description": "Type 2 diabetes mellitus with ketoacidosis without coma",
      "icd10_code": "E11.10",
      "reasoning": "Patient presents with classic triad of hyperglycemia (glucose 480), ketosis (ketones in urine), and metabolic acidosis (bicarbonate 12, anion gap 22, pH 7.21) in the setting of medication non-adherence. Symptoms of polydipsia, polyuria, and nausea are consistent with DKA.",
      "confidence": "High"
    },
    {
      "condition": "Community-acquired pneumonia",
      "icd10_description": "Pneumonia, unspecified organism",
      "icd10_code": "J18.9",
      "reasoning": "Chest X-ray shows right lower lobe infiltrate, and patient has fever and tachypnea. The pneumonia likely precipitated the DKA, as infections are common triggers.",
      "confidence": "High"
    },
    {
      "condition": "Type 2 diabetes mellitus, poorly controlled",
      "icd10_description": "Type 2 diabetes mellitus without complications",
      "icd10_code": "E11.9",
      "reasoning": "Underlying condition with medication non-adherence (ran out of metformin) that predisposed to DKA.",
      "confidence": "High"
    }
  ],
  "differential_diagnoses": [
    {
      "condition": "Sepsis",
      "icd10_description": "Sepsis, unspecified organism",
      "icd10_code": "A41.9",
      "reasoning": "Patient has fever, tachycardia, and pneumonia which could progress to sepsis. However, current presentation can be explained by DKA and pneumonia without invoking sepsis.",
      "confidence": "Medium"
    }
  ],
  "clinical_notes": "This patient has DKA precipitated by medication non-adherence and community-acquired pneumonia. Treatment should address both the metabolic derangement and the infection. Blood cultures and further workup for the pneumonia are warranted."
}

Example 2 (Primary Diagnosis with Multiple Differentials):
Vignette: "A 67-year-old female presents with acute onset right-sided facial droop, slurred speech, and left arm weakness that began 45 minutes ago. PMH notable for hypertension, hyperlipidemia, and paroxysmal atrial fibrillation on warfarin with inconsistent monitoring. Last INR was 1.8 two months ago. Vitals: BP 178/92, HR 88, RR 16, T 37.0°C. Neurological exam confirms right facial droop, dysarthria, and left arm drift. NIHSS score is 7."

Response:
{
  "primary_diagnoses": [
    {
      "condition": "Acute ischemic stroke",
      "icd10_description": "Cerebral infarction due to embolism of cerebral arteries",
      "icd10_code": "I63.4",
      "reasoning": "Patient presents with sudden-onset focal neurological deficits (facial droop, slurred speech, arm weakness) consistent with stroke. Risk factors include hypertension, hyperlipidemia, and especially atrial fibrillation with subtherapeutic anticoagulation (INR 1.8), suggesting a cardioembolic etiology.",
      "confidence": "High"
    }
  ],
  "differential_diagnoses": [
    {
      "condition": "Transient ischemic attack",
      "icd10_description": "Transient cerebral ischemic attack, unspecified",
      "icd10_code": "G45.9",
      "reasoning": "If symptoms resolve completely within 24 hours without evidence of infarction on imaging, this would be classified as a TIA rather than a stroke.",
      "confidence": "Medium"
    },
    {
      "condition": "Intracranial hemorrhage",
      "icd10_description": "Nontraumatic intracerebral hemorrhage, unspecified",
      "icd10_code": "I61.9",
      "reasoning": "Patient is on warfarin which increases risk of hemorrhagic stroke.  presentation can be similar to ischemic stroke. Would need neuroimaging to definitively rule out.",
      "confidence": "Medium"
    },
    {
      "condition": "Todd's paralysis post seizure",
      "icd10_description": "Postictal paralysis",
      "icd10_code": "G83.8",
      "reasoning": "Can present with transient unilateral weakness, though typically there would be a history of seizure activity preceding the deficits, which is not mentioned here.",
      "confidence": "Low"
    }
  ],
  "clinical_notes": "This is a case requiring urgent assessment for acute stroke intervention. The patient is within the time window for thrombolysis, but warfarin use complicates this decision. Immediate CT brain and measurement of current INR are essential. Neurology consultation for potential thrombolysis or endovascular intervention is indicated."
}
Only return a single string that can be parsed as .json. Do NOT return any additions of markdown or other modifiers. DO NOT any other additional content outside of a single .json string
"""

DDX_VALIDATION_SYSTEM = """You are a medical expert system tasked with analyzing clinical information, validating diagnoses, and providing structured output. You will be given a clinical vignette, a proposed diagnosis list, and a dictionary of clinical information. Your task is to analyze this information carefully and provide a structured assessment.

Given:
1. A clinical vignette describing a patient case
2. The diagnoses of the clinical vignette, and other differentials to be considered.
3. A dictionary of structured clinical information extracted from the case

Your tasks:

1. VALIDATION: Carefully evaluate if the proposed diagnoses are accurate, clinically sound, and fully supported by the information in the vignette and clinical data dictionary.
2. ALTERNATIVE DIAGNOSES: Evaluate provided differentials, if any, and also identify any other additional, potential acceptable diagnoses that fit the clinical picture based STRICTLY on the provided information. Do NOT suggest diagnoses that require additional information not present in the vignette or data dictionary.
3. ICD-10 CODING: Transform the confirmed diagnoses into appropriate ICD-10 codes, linking each diagnosis to the specific clinical findings that support it.

IMPORTANT CONSTRAINTS:
- You must ONLY reference keys that exist in the original clinical information dictionary
- You must NOT hallucinate or invent any clinical findings not explicitly stated
- Be EXTREMELY conservative when suggesting alternative diagnoses - only include those that are strongly supported by the provided information
- If there are no other reasonable alternative diagnoses, clearly state this
- Provide clear reasoning for any alternative diagnoses you suggest

EXAMPLES:
GOOD EXAMPLE:
Clinical information includes: {'history:symptoms:chest_pain': 'Severe chest pain', 'history:symptoms:radiation_to_left_arm': 'Pain radiating to left arm'}
Proposed diagnosis: ["Acute myocardial infarction"]
Differentials: ["Stable Angina"]

Response:
{
  "confirmed_diagnoses": [
    {
      "name": "Acute myocardial infarction",
      "icd_10": "I21.3",
      "relevant_keys": ["history:symptoms:chest_pain", "history:symptoms:radiation_to_left_arm"]
    }
  ],
  "other_acceptable_diagnoses": [
    {
      "name": "Stable Angina",
      "icd_10": "I20.9",
      "relevant_keys": ["history:symptoms:chest_pain", "history:symptoms:radiation_to_left_arm"],
      "reasoning": "Chest pain with radiation to the left arm can also be consistent with stable angina. Without further confirmatory tests, stable angina would remain a reasonable differential."
    }
  ]
}


GOOD EXAMPLE:
Clinical information includes: {'history:symptoms:chest_pain': 'Severe chest pain', 'history:symptoms:radiation_to_left_arm': 'Pain radiating to left arm', 'investigations:blood:troponin': 'Elevated', 'investigations:ecg:st_elevation': 'Present in V1-V4'}
Proposed diagnosis: ["Acute myocardial infarction"]
Differentials: []

Response:
{
  "confirmed_diagnoses": [
    {
      "name": "Acute myocardial infarction",
      "icd_10": "I21.3",
      "relevant_keys": ["history:symptoms:chest_pain", "history:symptoms:radiation_to_left_arm", "investigations:blood:troponin", "investigations:ecg:st_elevation"]
    }
  ],
  "other_acceptable_diagnoses": []
}

BAD EXAMPLE:
Clinical information includes: {'history:symptoms:chest_pain': 'Severe chest pain', 'history:symptoms:radiation_to_left_arm': 'Pain radiating to left arm', 'investigations:blood:troponin': 'Elevated', 'investigations:ecg:st_elevation': 'Present in V1-V4'}
Proposed diagnosis: ["Acute myocardial infarction"]
Differentials: []

Response:
{
  "confirmed_diagnoses": [
    {
      "name": "Acute myocardial infarction",
      "icd_10": "I21.3",
      "relevant_keys": ["history:symptoms:chest_pain", "history:symptoms:radiation_to_left_arm", "investigations:blood:troponin", "investigations:ecg:st_elevation"]
    }
  ],
  "other_acceptable_diagnoses": [
    {
      "name": "Pericarditis",
      "icd_10": "I30.9",
      "relevant_keys": ["history:symptoms:chest_pain", "investigations:ecg:st_elevation"],
      "reasoning": "Pericarditis can present with chest pain and ECG changes"
    },
    {
      "name": "Pneumonia",
      "icd_10": "J18.9",
      "relevant_keys": ["history:symptoms:chest_pain", "history:symptoms:fever"],
      "reasoning": "Pneumonia can present with chest pain and fever"
    }
  ]
}
(This is bad because it suggests pneumonia despite no fever being documented in the clinical information, and pericarditis without sufficient supporting evidence)

GOOD EXAMPLE:
Clinical information includes: {'history:symptoms:abdominal_pain': 'Severe abdominal pain', 'physical:abdomen:tenderness': 'Right lower quadrant tenderness', 'investigations:blood:wbc': 'Elevated', 'investigations:imaging:ct_scan': 'Appendiceal inflammation'}
Proposed diagnosis: ["Appendicitis"]
Differentials: []

Response:
{
  "confirmed_diagnoses": [
    {
      "name": "Acute appendicitis",
      "icd_10": "K35.80",
      "relevant_keys": ["history:symptoms:abdominal_pain", "physical:abdomen:tenderness", "investigations:blood:wbc", "investigations:imaging:ct_scan"]
    }
  ],
  "other_acceptable_diagnoses": []
}

BAD EXAMPLE:
Clinical information includes: {'history:symptoms:abdominal_pain': 'Severe abdominal pain', 'physical:abdomen:tenderness': 'Right lower quadrant tenderness', 'investigations:blood:wbc': 'Elevated', 'investigations:imaging:ct_scan': 'Appendiceal inflammation'}
Proposed diagnosis: ["Appendicitis", "Gastroenteritis"]
Differentials: []

Response:
{
  "confirmed_diagnoses": [
    {
      "name": "Acute appendicitis",
      "icd_10": "K35.80",
      "relevant_keys": ["history:symptoms:abdominal_pain", "physical:abdomen:tenderness", "investigations:blood:wbc", "investigations:imaging:ct_scan"]
    },
    {
      "name": "Gastroenteritis",
      "icd_10": "A09",
      "relevant_keys": ["history:symptoms:abdominal_pain", "history:symptoms:diarrhea", "history:symptoms:vomiting"]
    }
  ],
  "other_acceptable_diagnoses": []
}
(This is bad because it confirms gastroenteritis despite no documentation of diarrhea or vomiting in the clinical information)

Return your analysis in the following JSON format. Return nothing but pure .json:
{
  "confirmed_diagnoses": [
    {
      "name": "diagnosis name",
      "icd_10": "code",
      "relevant_keys": ["list", "of", "supporting", "keys", "from", "dictionary"]
    }
  ],
  "other_acceptable_diagnoses": [
    {
      "name": "alternative diagnosis name",
      "icd_10": "code",
      "relevant_keys": ["list", "of", "supporting", "keys"],
      "reasoning": "concise explanation of why this is a reasonable alternative"
    }
  ]
}

If there are no other acceptable diagnoses, return an empty list for "other_acceptable_diagnoses".
Only return a single string that can be parsed as .json. Do NOT return any additions of markdown or other modifiers. DO NOT any other additional content outside of a single .json string. 
"""

DDX_VALIDATION_PROMPT = """Vignette: {vignette}\nProposed Diagnosis: {ddx}\nDifferentials: {differentials}\nStructured Clinical Information: {clin_dict}"""
