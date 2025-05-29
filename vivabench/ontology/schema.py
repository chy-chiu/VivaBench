from collections import defaultdict
from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, List, Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from vivabench.ontology.templates import FULL_CASE_TEMPLATE, _symptom_description
from vivabench.utils import normalize_key, prettify


class ClinicalData(BaseModel):

    def __getitem__(self, idx):
        return self.__dict__.get(idx)

    def get(self, key, default=""):
        """Retrieves an attribute value, returning default if not found.
        Args:
            key: The attribute name to retrieve
            default: Value to return if attribute doesn't exist
        Returns:
            The attribute value or default
        """
        return getattr(self, key, default)


class Symptom(ClinicalData):
    """
    Represents a clinical symptom with detailed attributes for medical documentation.

    This class captures comprehensive information about a patient's symptom,
    including its presence, temporal characteristics, location, and modifying factors.
    """

    name: str  # The name of the symptom
    present: bool = (
        True  # Whether the symptom is present (True) or explicitly denied (False)
    )

    # Temporal characteristics
    # When the symptom first began (e.g., "2 days ago", "gradually over weeks")
    onset: Optional[str] = None
    # How long the symptom has persisted (e.g., "3 hours", "intermittent for 2 weeks")
    duration: Optional[str] = None
    # How the symptom has evolved over time (e.g., "worsening", "improving", "stable")
    progression: Optional[str] = None
    # When the symptom occurs (e.g., "morning", "after meals", "during exercise")
    timing: Optional[str] = None

    # Localization and characterization
    # Intensity of the symptom (e.g., "mild", "moderate", "severe")
    severity: Optional[str] = None
    # Body system affected (e.g., "cardiovascular", "respiratory")
    system: Optional[str] = None
    # Anatomical location of the symptom (e.g., "left lower quadrant", "behind sternum")
    location: Optional[str] = None
    # Quality or nature of the symptom (e.g., "sharp", "dull", "throbbing")
    character: Optional[str] = None
    # Whether and where the symptom spreads (e.g., "radiates to left arm")
    radiation: Optional[str] = None

    # Modifying factors
    # Factors that improve the symptom (e.g., "rest", "medication")
    alleviating_factors: List[str] = Field(default_factory=list)
    # Factors that worsen the symptom (e.g., "movement", "eating")
    aggravating_factors: List[str] = Field(default_factory=list)

    # Related information
    # Other symptoms that occur alongside this one (e.g., "nausea", "dizziness")
    associated_symptoms: List[str] = Field(default_factory=list)
    # Circumstances surrounding the symptom (e.g., "occurs after drinking alcohol")
    context: Optional[str] = None
    # Detailed narrative about this specific symptom's history
    history: Optional[str] = None

    ATTR_KEYS: ClassVar[List] = [
        "name",
        "present",
        "onset",
        "duration",
        "progression",
        "timing",
        "severity",
        "system",
        "location",
        "character",
        "radiation",
        "alleviating_factors",
        "aggravating_factors",
        "associated_symptoms",
        "timing",
        "context",
        "history",
    ]

    def keys(self):
        available_keys = []
        for k in self.ATTR_KEYS:
            if self.get(k):
                available_keys.append(k)
        return available_keys

    @property
    def prompt(self):
        """Returns a complete textual description of the symptom with all available details."""
        return _symptom_description(self, addit_keys=self.ATTR_KEYS)

    def get_prompt(self, addit_keys=[]):
        """Returns a textual description of the symptom with only the specified additional keys.
        Args:
            addit_keys: List of additional attribute keys to include in the description
        Returns:
            String description of the symptom with selected attributes
        """
        return _symptom_description(self, addit_keys=addit_keys)

    @property
    def bullet(self):
        """Returns a complete bullet-point formatted description of all symptom attributes."""
        return self.get_bullet()

    def get_bullet(self, addit_keys=None):
        """Returns a bullet-point formatted description of selected symptom attributes.
        Args:
            addit_keys: List of specific attributes to include. If None, includes all attributes.
        Returns:
            String with bullet points for the specified attributes (or all if None)
        """
        _display = f"## {prettify(self.name)}\n"

        requested_keys = self.ATTR_KEYS if addit_keys is None else addit_keys
        keys_to_display = set()

        for k in requested_keys:
            v = self.get(k, "")
            # We don't care about name or system
            if k in ["name", "system"]:
                continue
            # Only show present if it's a relevant negative
            if k == "present" and not v:
                keys_to_display.add("present")
            # For temporal attributes, they are quite interchangeable. Therefore get any that is not None for routing.
            if k in (temporal_attr := ["onset", "duration", "progression", "timing"]):
                for t in temporal_attr:
                    if self.get(t):
                        keys_to_display.add(t)
            # Only show None values when specifically requested
            if not v and addit_keys:
                keys_to_display.add(k)
                if addit_keys is not None:
                    _display += f"- {prettify(k)}: None\n"
            elif v:
                keys_to_display.add(k)

        for k in self.ATTR_KEYS:
            if k in keys_to_display:
                v = self.get(k)
                if not v:
                    v = "None"
                _display += (
                    f"- {prettify(k)}: {', '.join(v) if isinstance(v, list) else v}\n"
                )
        return _display


class Demographics(ClinicalData):
    age: Optional[Union[int, str]] = None
    unit: Optional[str] = None
    gender: Optional[str] = ""
    ethnicity: Optional[str] = None
    place_of_birth: Optional[str] = None

    @property
    def prompt(self):
        return f"{self.age} {self.unit} old {self.gender}"

    @property
    def bullet(self):
        _display = "## Patient Information"
        for attr in [
            "age",
            "gender",
            "race",
            "ethnicity",
            "place_of_birth",
        ]:
            if v := self.get(attr, ""):
                _display += f"- {prettify(attr)}: {v}\n"
        return _display


class Medication(ClinicalData):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    indication: Optional[str] = None
    current: bool = True

    @property
    def prompt(self) -> str:
        _prompt = f"{self.name}"
        for k in ["dosage", "route", "frequency"]:
            if self[k]:
                _prompt += " " + self[k]
        return _prompt

    @property
    def bullet(self):
        _display = self.prompt
        if self.current:
            _display += f"\nCurrent: {self.current}"
        if self.indication:
            _display += f"\nIndication: {self.indication}"
        return _display


class Allergy(ClinicalData):
    allergen: str
    reaction: Optional[str] = None
    severity: Optional[str] = None

    @property
    def prompt(self):
        _prompt = f"{self.allergen}"
        for k in ["reaction", "severity"]:
            if self[k]:
                _prompt += f" | {prettify(k)}: {prettify(self[k])}"
        return _prompt


class SocialHistory(ClinicalData):
    """Represents a patient's social history including lifestyle factors and living conditions."""

    # Smoking-related information
    # Whether the patient currently smokes
    smoking_current: Optional[bool] = None
    # Cumulative smoking exposure in pack-years
    smoking_pack_years: Optional[float] = None
    # Years since quitting smoking
    smoking_quit: Optional[int | str] = None

    # Substance use
    # Pattern and amount of alcohol consumption
    alcohol_use: Optional[str] = None
    # Use of recreational drugs or other substances
    substance_use: Optional[str] = None

    # Life circumstances
    # Patient's job or employment status
    occupation: Optional[str] = None
    # Housing status and who the patient lives with
    living_situation: Optional[str] = None
    # Recent or relevant travel
    travel_history: Optional[str] = None

    # Lifestyle factors
    # Physical activity patterns
    exercise: Optional[str] = None
    # Dietary habits and restrictions
    diet: Optional[str] = None
    # Sexual history and practices
    sexual: Optional[str] = None

    # Any other relevant social history items
    other: Optional[Dict[str, str]] = None

    ATTR_KEYS: ClassVar[List] = [
        "smoking_current",
        "smoking_pack_years",
        "smoking_quit",
        "alcohol_use",
        "substance_use",
        "occupation",
        "living_situation",
        "travel_history",
        "exercise",
        "diet",
        "sexual",
    ]

    @property
    def full_prompt(self):
        """
        Returns a complete textual description of the social history with all available details.
        Returns:
            String description of the patient's social history
        """
        _prompt = ""

        # Format smoking history
        if self.smoking_pack_years is not None:
            if self.smoking_current:
                _prompt += f"Current smoker, {self.smoking_pack_years} pack years smoking history.\n"
            else:
                _prompt += f"Ex-smoker, {self.smoking_pack_years} pack years smoking history.\n"
                if self.smoking_quit:
                    _prompt += f"Quit {self.smoking_quit} years ago.\n"
        elif self.smoking_current is not None:
            if self.smoking_current:
                _prompt += "Current smoker, pack years unknown.\n"
            else:
                _prompt += "Non-smoker.\n"

        # Add other social history elements
        for k in [
            "alcohol_use",
            "substance_use",
            "occupation",
            "living_situation",
            "travel_history",
            "exercise",
            "diet",
            "sexual",
        ]:
            if self.get(k):
                _prompt += f"{prettify(k)}: {self.get(k)}\n"

        # Add any additional items from the 'other' dictionary
        if self.other:
            for key, value in self.other.items():
                _prompt += f"{prettify(key)}: {value}\n"

        return _prompt

    def prompt(self, key: str):
        """
        Returns a specific element of the social history.
        Args:
            key: The specific social history element to retrieve
        Returns:
            String description of the requested element or None if not available
        """
        if "smoking" in key.lower():
            if self.smoking_pack_years is not None:
                if self.smoking_current:
                    return f"Smoking history: Current smoker, {self.smoking_pack_years} pack years smoking history."
                else:
                    quit_info = (
                        f" Quit {self.smoking_quit} years ago."
                        if self.smoking_quit
                        else ""
                    )
                    return f"Smoking history: Ex-smoker, {self.smoking_pack_years} pack years smoking history.{quit_info}"
            elif self.smoking_current is not None:
                return (
                    "Smoking history: Current smoker, pack years unknown."
                    if self.smoking_current
                    else "Smoking history: Non-smoker"
                )
            else:
                return "Smoking history: Not documented"
        elif v := self.get(key):
            return f"{prettify(key)}: {prettify(v)}"
        else:
            return None

    @property
    def full_bullet(self):
        """
        Returns a complete bullet-point formatted description of all social history elements.

        Returns:
            String with bullet points for all non-empty attributes
        """
        return self.bullet()

    def bullet(self, keys=None):
        """
        Returns a bullet-point formatted description of selected social history elements.

        Args:
            addit_keys: List of specific elements to include. If None, includes all elements.

        Returns:
            String with bullet points for the specified elements (or all if None)
        """
        _display = "## Social History\n"

        all_keys = self.keys()
        keys_to_display = all_keys if keys is None else keys

        # Handle smoking information specially
        if any(k for k in keys_to_display if "smoking" in k):
            if self.smoking_pack_years is not None:
                if self.smoking_current:
                    _display += f"- Smoking: Current smoker, {self.smoking_pack_years} pack years\n"
                else:
                    quit_info = (
                        f", quit {self.smoking_quit} years ago"
                        if self.smoking_quit
                        else ""
                    )
                    _display += f"- Smoking: Ex-smoker, {self.smoking_pack_years} pack years{quit_info}\n"
            elif self.smoking_current is not None:
                status = "Current smoker" if self.smoking_current else "Non-smoker"
                _display += f"- Smoking: {status}\n"
            elif keys is not None:
                _display += "- Smoking: None\n"

        # Add other elements
        for k in [k for k in keys_to_display if "smoking" not in k]:
            v = self.get(k)
            if v is None:
                if (
                    keys is not None
                ):  # Only show None values when specifically requested
                    _display += f"- {prettify(k)}: None\n"
            elif v:  # Skip empty strings and collections
                _display += f"- {prettify(k)}: {v}\n"

        return _display

    def keys(self):
        """
        Returns a list of all the social history elements.
        Returns:
            List of attribute keys
        """
        _keys = []
        for key in self.ATTR_KEYS:
            if self.get(key):
                _keys.append(key)
        if self.get("other") and isinstance(self.get("other"), dict):
            _keys.extend(list(self.other.keys()))

        return _keys


class FamilyHistoryItem(ClinicalData):
    """
    Represents a single condition in a patient's family history.

    This class captures information about a medical condition affecting
    a family member, including their relationship to the patient and
    relevant details about the condition.
    """

    # The medical condition or diagnosis
    condition: str
    # The family relationship to the patient (e.g., "mother", "brother")
    relationship: Optional[str] = None
    # Age when the family member developed the condition
    age_at_onset: Optional[int] = None
    # Additional relevant information about this condition
    notes: Optional[str] = None

    @property
    def prompt(self):
        """
        Returns a textual description of this family history item.

        Returns:
            String description of the family history item
        """
        if not self.relationship:
            _prompt = f"No family history of {self.condition}"
        else:
            _prompt = f"{self.relationship.capitalize()} - {self.condition}"
            if self.age_at_onset:
                _prompt += f". Happened at age {self.age_at_onset}. "
            if self.notes:
                _prompt += " - " + self.notes
        return _prompt


class PastMedicalHistoryItem(ClinicalData):
    """
    Represents a single condition in a patient's past medical history.

    This class captures ianformation about a medical condition the patient
    has experienced, including whether it's ongoing and additional details.
    """

    condition: str  # The medical condition or diagnosis
    present: bool  # Whether the condition is confirmed (True) or ruled out (False)
    ongoing: Optional[bool] = (
        True  # Whether the condition is current/active or resolved
    )
    description: Optional[str] = None  # Additional details about the condition

    @property
    def prompt(self):
        """
        Returns a textual description of this past medical history item.

        Returns:
            String description of the medical history item
        """
        if not self.present:
            return f"No history of {self.condition}."

        status = "Has" if self.ongoing else "Had"
        _prompt = f"{status} {self.condition}. "

        if self.description:
            _prompt += self.description

        return _prompt


class History(ClinicalData):
    chief_complaint: str
    symptoms: Dict[str, Symptom]
    past_medical_history: Dict[str, PastMedicalHistoryItem] = Field(
        default_factory=dict
    )
    medications: Optional[List[Medication]] = Field(default_factory=list)
    allergies: Optional[List[Allergy]] = Field(default_factory=list)
    social_history: Optional[SocialHistory] = None
    family_history: Dict[str, FamilyHistoryItem] = Field(default_factory=dict)

    def keys(self):
        hx_keys = []
        hx_keys.extend([f"symptoms:{k}" for k in self.symptoms.keys()])
        if self.social_history:
            hx_keys.extend([f"social_history:{k}" for k in self.social_history.keys()])
        hx_keys.extend(
            [f"past_medical_history:{k}" for k in self.past_medical_history.keys()]
        )
        hx_keys.extend([f"family_history:{k}" for k in self.family_history.keys()])

        for k in ["past_medical_history", "family_history", "allergies", "medications"]:
            if self[k]:
                hx_keys.append(k)

        return hx_keys

    def dict(self):
        hx_items = {}

        # Handle symptoms, social_history, past_medical_history, and family_history
        for symptom_key, symptom in self.symptoms.items():
            hx_items[f"symptoms:{symptom_key}"] = symptom.name

        if self.social_history:
            for social_key in self.social_history.keys():
                hx_items[f"social_history:{social_key}"] = str(
                    self.social_history.get(social_key)
                )

        for pmh_key, pmh_item in self.past_medical_history.items():
            hx_items[f"past_medical_history:{pmh_key}"] = pmh_item.condition

        for fh_key, fh_item in self.family_history.items():
            hx_items[f"family_history:{fh_key}"] = fh_item.condition

        # Handle medications, allergies as list
        if self.medications:
            hx_items["medications"] = self.medication_list

        if self.allergies:
            hx_items["allergies"] = self.allergies_list

        return hx_items

    @property
    def symptom_list(self):
        _symptom_dict = defaultdict(list)
        for symptom in self.symptoms.values():
            if symptom.present:
                _symptom_dict[symptom.system].append(symptom.bullet)
        _prompt = ""
        for system, symptoms in _symptom_dict.items():
            _prompt += f"### {prettify(system)}\n"
            for symptom in symptoms:
                _prompt += symptom

        return _prompt

    @property
    def pmh_list(self):
        if not self.past_medical_history:
            "Nil significant medical history"
        _pmh_list = ""
        for pmh in self.past_medical_history.values():
            # Only return positives, not relevant negatives
            if pmh.present and pmh.ongoing:
                _pmh_list += f"- {pmh.condition}\n"
        return _pmh_list if _pmh_list else "Nil significant medical history"

    @property
    def medication_list(self):
        if not self.medications:
            return "Nil medications"
        medications = [
            f"- {medication.prompt}"
            for medication in self.medications
            if medication.current
        ]
        if medications:
            return "\n".join(medications)
        else:
            return "Nil medications"

    @property
    def allergies_list(self):
        if not self.allergies:
            return "No known allergies"
        allergies = [f"- {allergen.prompt}" for allergen in self.allergies]
        if allergies:
            return "\n".join(allergies)
        else:
            return "No known allergies"

    @property
    def social_history_list(self):
        return (
            self.social_history.full_prompt
            if self.social_history
            else "Nil significant"
        )

    @property
    def family_history_list(self):
        if not self.family_history:
            return "No significant family history"
        fh_list = [f"- {fh.prompt}" for fh in self.family_history.values()]
        if fh_list:
            return "\n".join(fh_list)
        else:
            return "No significant family history"


class Vitals(ClinicalData):
    heart_rate: Optional[Any] = None
    blood_pressure_systolic: Optional[Any] = None
    blood_pressure_diastolic: Optional[Any] = None
    temperature: Optional[Any] = None
    respiratory_rate: Optional[Any] = None
    oxygen_saturation: Optional[Any] = None
    pain_score: Optional[Any] = None
    height: Optional[Any] = None
    weight: Optional[Any] = None
    bmi: Optional[Any] = None
    gcs: Optional[Any] = None

    # Store temporal notes for each vital sign
    temporal_notes: Optional[Dict[str, List[str]]] = Field(default_factory=dict)

    @model_validator(mode="before")
    def normalize_vitals(cls, data):
        if not isinstance(data, dict):
            return data

        # For each vital sign that might be a list or single value
        for field in [
            "heart_rate",
            "blood_pressure_systolic",
            "blood_pressure_diastolic",
            "temperature",
            "respiratory_rate",
            "oxygen_saturation",
            "pain_score",
            "height",
            "weight",
            "bmi",
            "gcs",
        ]:
            # Skip if field is not present
            if field not in data:
                continue

            # Try to convert string values to appropriate numeric types
            if isinstance(data[field], str):
                try:
                    # Try to convert to int first (for heart rate, BP, etc.)
                    data[field] = int(data[field])
                except ValueError:
                    try:
                        # Try to convert to float (for temp, O2 sat, etc.)
                        data[field] = float(data[field])
                    except ValueError:
                        # Keep as string if it can't be converted (descriptive terms)
                        pass

            # Handle list of values - convert each element if possible
            elif isinstance(data[field], list):
                converted_values = []
                for value in data[field]:
                    if isinstance(value, str):
                        try:
                            # Try int first
                            converted_values.append(int(value))
                        except ValueError:
                            try:
                                # Try float next
                                converted_values.append(float(value))
                            except ValueError:
                                # Keep as string if can't convert
                                converted_values.append(value)
                    else:
                        converted_values.append(value)
                data[field] = converted_values

            # If it's not already a list but we have multiple values in systems.vitals
            # and top-level vitals, convert to a list
            if not isinstance(data[field], list):
                # Check if we need to merge with another value
                systems_vitals = None
                if (
                    "systems" in data
                    and "vitals" in data["systems"]
                    and field in data["systems"]["vitals"]
                ):
                    systems_vitals = data["systems"]["vitals"][field]

                    # Try to convert systems_vitals if it's a string
                    if isinstance(systems_vitals, str):
                        try:
                            systems_vitals = int(systems_vitals)
                        except ValueError:
                            try:
                                systems_vitals = float(systems_vitals)
                            except ValueError:
                                pass

                if systems_vitals is not None and systems_vitals != data[field]:
                    # Create a list with both values
                    data[field] = [data[field], systems_vitals]

                    # Initialize temporal_notes if not present
                    if "temporal_notes" not in data:
                        data["temporal_notes"] = {}

                    # Add a note about the temporal relationship if we can determine it
                    if "temporal_notes" in data and field not in data["temporal_notes"]:
                        data["temporal_notes"][field] = ["initial", "later"]

        # Remove the duplicate vitals from systems if present
        if "systems" in data and "vitals" in data["systems"]:
            del data["systems"]["vitals"]

        return data

    @property
    def prompt(self):
        return self.initial_prompt()

    def initial_prompt(self) -> str:
        """Returns a string representation of the initial vital signs."""
        parts = []

        # For each vital sign, get the initial value (first if list)
        vital_mappings = {
            "temperature": "Temperature",
            "heart_rate": "HR",
            "blood_pressure_systolic": "BP",
            "blood_pressure_diastolic": "",  # Will be combined with systolic
            "respiratory_rate": "RR",
            "oxygen_saturation": "O2 sat",
            "pain_score": "Pain",
            "gcs": "GCS",
        }

        for field, label in vital_mappings.items():
            value = getattr(self, field)

            # Skip if not present
            if value is None:
                continue

            # Get initial value if it's a list
            initial_value = value[0] if isinstance(value, list) else value

            # Special handling for blood pressure
            if (
                field == "blood_pressure_systolic"
                and self.blood_pressure_diastolic is not None
            ):
                diastolic = self.blood_pressure_diastolic
                diastolic_value = (
                    diastolic[0] if isinstance(diastolic, list) else diastolic
                )
                parts.append(f"BP {initial_value}/{diastolic_value} mmHg")
                continue
            elif field == "blood_pressure_diastolic":
                # Skip as it's handled with systolic
                continue

            # Add units based on the field
            if field == "temperature":
                parts.append(f"{label} {initial_value}°C")
            elif field == "heart_rate":
                parts.append(f"{label} {initial_value} bpm")
            elif field == "respiratory_rate":
                parts.append(f"{label} {initial_value}/min")
            elif field == "oxygen_saturation":
                parts.append(f"{label} {initial_value}%")
            else:
                parts.append(f"{label} {initial_value}")

        return ", ".join(parts)

    def full_prompt(self) -> str:
        """Returns a string representation of all vital signs with their trajectories."""
        parts = []

        # For each vital sign, format its trajectory
        vital_mappings = {
            "temperature": "Temperature",
            "heart_rate": "HR",
            "blood_pressure_systolic": "BP",
            "blood_pressure_diastolic": "",  # Will be combined with systolic
            "respiratory_rate": "RR",
            "oxygen_saturation": "O2 sat",
            "pain_score": "Pain",
            "gcs": "GCS",
        }

        for field, label in vital_mappings.items():
            value = getattr(self, field)

            # Skip if not present
            if value is None:
                continue

            # Special handling for blood pressure
            if (
                field == "blood_pressure_systolic"
                and self.blood_pressure_diastolic is not None
            ):
                diastolic = self.blood_pressure_diastolic

                # If both are lists of the same length
                if (
                    isinstance(value, list)
                    and isinstance(diastolic, list)
                    and len(value) == len(diastolic)
                ):
                    bp_values = [f"{s}/{d} mmHg" for s, d in zip(value, diastolic)]

                    # Add temporal notes if available
                    notes = ""
                    if field in self.temporal_notes:
                        notes_list = [
                            f" ({note})" for note in self.temporal_notes[field]
                        ]
                        bp_trajectory = " → ".join(
                            [f"{bp}{note}" for bp, note in zip(bp_values, notes_list)]
                        )
                    else:
                        bp_trajectory = " → ".join(bp_values)

                    parts.append(f"BP: {bp_trajectory}")
                else:
                    # Handle mixed types or different lengths
                    systolic_value = value[0] if isinstance(value, list) else value
                    diastolic_value = (
                        diastolic[0] if isinstance(diastolic, list) else diastolic
                    )
                    parts.append(f"BP {systolic_value}/{diastolic_value} mmHg")

                continue
            elif field == "blood_pressure_diastolic":
                # Skip as it's handled with systolic
                continue

            # Format based on whether it's a list or single value
            if isinstance(value, list):
                # Add units based on the field
                if field == "temperature":
                    values_with_units = [f"{v}°C" for v in value]
                elif field == "heart_rate":
                    values_with_units = [f"{v} bpm" for v in value]
                elif field == "respiratory_rate":
                    values_with_units = [f"{v}/min" for v in value]
                elif field == "oxygen_saturation":
                    values_with_units = [f"{v}%" for v in value]
                else:
                    values_with_units = [str(v) for v in value]

                # Add temporal notes if available
                if field in self.temporal_notes:
                    notes_list = [f" ({note})" for note in self.temporal_notes[field]]
                    trajectory = " → ".join(
                        [
                            f"{val}{note}"
                            for val, note in zip(values_with_units, notes_list)
                        ]
                    )
                else:
                    trajectory = " → ".join(values_with_units)

                parts.append(f"{label}: {trajectory}")
            else:
                # Single value
                if field == "temperature":
                    parts.append(f"{label} {value}°C")
                elif field == "heart_rate":
                    parts.append(f"{label} {value} bpm")
                elif field == "respiratory_rate":
                    parts.append(f"{label} {value}/min")
                elif field == "oxygen_saturation":
                    parts.append(f"{label} {value}%")
                else:
                    parts.append(f"{label} {value}")

        return ", ".join(parts)


class PhysicalFinding(ClinicalData):
    name: str
    description: str
    augmented: bool = False

    @property
    def prompt(self):
        desc = prettify(self.description)
        if not desc.endswith("."):
            desc += "."
        return f"{prettify(self.name)}: {desc} "


class PhysicalExamination(ClinicalData):
    vitals: Vitals = Field(default_factory=Vitals)
    systems: Dict[str, Dict[str, PhysicalFinding]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_keys(self):

        if self.systems:
            _systems = {}
            for system, findings in self.systems.items():
                _systems[system] = {normalize_key(k): v for k, v in findings.items()}
            self.systems = _systems
        return self

    # TODO: Fix this later to be retrievable in pieces, but this will do for now
    @property
    def prompt(self):
        _prompt = str(self.vitals.prompt) + "\n"

        for k, v in self.systems.items():
            _prompt += f"{prettify(k)}: "
            findings = []
            for _v in v.values():
                if isinstance(_v, PhysicalFinding):
                    findings.append(_v.prompt)
                else:
                    findings.append(str(_v))
            _prompt += " ".join(findings) + "\n"
        return _prompt

    def keys(self):
        keys = []
        for system, findings in self.systems.items():
            if findings.values():
                keys.extend([f"{system}:{finding}" for finding in findings.keys()])

        return keys

    def dict(self):
        _items = {"vitals": self.vitals.prompt}
        for k in self.keys():
            _items[k] = self.get_prompt(k)

        return _items

    def get_prompt(self, query: str):
        if ":" in query:
            request_system, request_exam = query.split(":")

            if system := self.systems.get(request_system):
                if exam := system.get(request_exam):
                    return exam.prompt
        return self.get_default(query)

    def get_default(self, query: str):
        if ":" in query:
            _, request_exam = query.split(":")
            return f"{prettify(request_exam)} - Negative."
        elif "vitals" in query:
            return self.vitals.prompt.replace("\n", " ")
        else:
            # TODO: Make default examination for each system
            return ["Unremarkable"]


class InvestigationResult(ClinicalData):
    name: str
    value: Any
    units: Optional[Any] = None
    reference_range: Optional[str] = None
    flag: Optional[str] = None
    note: Optional[str] = None
    specimen_type: Optional[Any] = None

    @model_validator(mode="before")
    def convert_and_normalize(cls, data):
        # Handle case where data is a list of dictionaries
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Combine the list into a single dictionary
            combined_data = {}

            # Extract name from first item if available
            if "name" in data[0]:
                combined_data["name"] = data[0]["name"]

            # Combine values from all items
            combined_values = []
            for item in data:
                if "name" in item and "value" in item:
                    component_name = item["name"]
                    component_value = item["value"]
                    units = (
                        f" {item['units']}" if "units" in item and item["units"] else ""
                    )
                    combined_values.append(
                        f"{component_name}: {component_value}{units}"
                    )

            combined_data["value"] = ", ".join(combined_values)

            # Add any notes if present
            notes = [
                item.get("note") for item in data if "note" in item and item["note"]
            ]
            if notes:
                combined_data["note"] = "; ".join(notes)

            # Use the combined data for further processing
            data = combined_data

        # Continue with normal processing for dictionary data
        if isinstance(data, dict):
            # Process the value field if it exists
            if "value" in data:
                value = data["value"]

                # Try to convert string value to numeric if possible
                if isinstance(value, str):
                    try:
                        # Try to convert to float first
                        numeric_value = float(value)
                        # If it's a whole number, convert to int
                        if numeric_value.is_integer():
                            data["value"] = int(numeric_value)
                        else:
                            data["value"] = numeric_value
                    except ValueError:
                        # Keep as string if it can't be converted
                        pass

                # Handle dictionary values
                elif isinstance(value, dict):
                    # Try to convert each value in the dictionary
                    for k, v in value.items():
                        if isinstance(v, str):
                            try:
                                numeric_v = float(v)
                                if numeric_v.is_integer():
                                    value[k] = int(numeric_v)
                                else:
                                    value[k] = numeric_v
                            except ValueError:
                                pass

                    # Convert dict to a readable string format
                    value_parts = []
                    for k, v in value.items():
                        # Format each key-value pair
                        k_formatted = k.replace("_", " ").title()
                        v_formatted = str(v) if v is not None else "None"
                        value_parts.append(f"{k_formatted}: {v_formatted}")

                    # Join all parts with commas
                    data["value"] = ", ".join(value_parts)

                # Handle list of values
                elif isinstance(value, list):
                    # Try to convert each value in the list
                    converted_values = []
                    for v in value:
                        if isinstance(v, str):
                            try:
                                numeric_v = float(v)
                                if numeric_v.is_integer():
                                    converted_values.append(int(numeric_v))
                                else:
                                    converted_values.append(numeric_v)
                            except ValueError:
                                converted_values.append(v)
                        else:
                            converted_values.append(v)

                    data["value"] = " → ".join([str(v) for v in converted_values])

            # Process the units field if it exists and is a dict
            if "units" in data and isinstance(data["units"], dict):
                # For units, we'll take the most relevant unit or combine them
                units_values = [v for v in data["units"].values() if v]
                if units_values:
                    data["units"] = ", ".join(units_values)
                else:
                    data["units"] = None

            # Handle specimen_type as a list or dict
            if "specimen_type" in data:
                if isinstance(data["specimen_type"], list):
                    data["specimen_type"] = ", ".join(
                        [str(s) for s in data["specimen_type"]]
                    )
                elif isinstance(data["specimen_type"], dict):
                    specimen_values = [v for v in data["specimen_type"].values() if v]
                    if specimen_values:
                        data["specimen_type"] = ", ".join(
                            [str(s) for s in specimen_values]
                        )
                    else:
                        data["specimen_type"] = None

        return data

    @property
    def prompt(self) -> str:
        unit = self.units or ""
        reference_range = f"({self.reference_range})" if self.reference_range else ""

        value_str = str(self.value)

        # Include note if available
        note_str = f" - {self.note}" if self.note else ""

        return f"- {self.name}: {value_str} {unit} {reference_range}{note_str}"


class ImagingStudy(ClinicalData):
    image_type: str  # e.g., "X-ray", "CT"
    region: str  # e.g. Chest, Abdomen
    findings: str
    impression: str
    recommendation: Optional[str] = None


class Investigations(ClinicalData):
    bedside: Dict[
        str, Union[InvestigationResult, str, List[Union[InvestigationResult, Dict]]]
    ] = Field(
        default_factory=dict
    )  # For any bedside tests such as ECG
    blood: Dict[
        str, Union[InvestigationResult, str, List[Union[InvestigationResult, Dict]]]
    ] = Field(
        default_factory=dict
    )  # Any blood / serological testing
    urine: Dict[
        str, Union[InvestigationResult, str, List[Union[InvestigationResult, Dict]]]
    ] = Field(
        default_factory=dict
    )  # Any urine testing, such as urine white cell count
    csf: Dict[
        str, Union[InvestigationResult, str, List[Union[InvestigationResult, Dict]]]
    ] = Field(
        default_factory=dict
    )  # Any testing involving cerebrospinal fluid such as lumbar puncture
    other_fluid: Dict[
        str, Union[InvestigationResult, str, List[Union[InvestigationResult, Dict]]]
    ] = Field(
        default_factory=dict
    )  # Any testing involving any other extracted fluid, such as joint aspirate, ascites tap
    microbiology: Dict[
        str, Union[InvestigationResult, str, List[Union[InvestigationResult, Dict]]]
    ] = Field(
        default_factory=dict
    )  # Any microbiology testing, such as sputum culture
    genetic: Dict[
        str, Union[InvestigationResult, str, List[Union[InvestigationResult, Dict]]]
    ] = Field(
        default_factory=dict
    )  # For genetic testing results in particular
    tissue: Dict[
        str, Union[InvestigationResult, str, List[Union[InvestigationResult, Dict]]]
    ] = Field(
        default_factory=dict
    )  # For any tissue samples, e.g. biopsy
    other: Dict[
        str, Union[InvestigationResult, str, List[Union[InvestigationResult, Dict]]]
    ] = Field(
        default_factory=dict
    )  # For any other special tests, such as lung function test

    @model_validator(mode="before")
    def normalize_investigation_results(cls, data):
        if not isinstance(data, dict):
            return data

        # Process each category of investigations
        for category in [
            "bedside",
            "blood",
            "urine",
            "csf",
            "other_fluid",
            "microbiology",
            "genetic",
            "tissue",
            "other",
        ]:
            if category not in data:
                continue

            # Process each investigation in this category
            for test_name, test_result in list(data[category].items()):
                # Handle case where test_result is a list of dictionaries
                if isinstance(test_result, list) and all(
                    isinstance(item, dict) for item in test_result
                ):
                    # If it's a list with multiple components of the same test
                    # Combine them into a single result
                    combined_result = {}

                    # Extract name from first item if available
                    if "name" in test_result[0]:
                        combined_result["name"] = test_result[0]["name"]

                    # Combine values from all items
                    combined_values = []
                    for item in test_result:
                        if "name" in item and "value" in item:
                            component_name = item["name"]
                            component_value = item["value"]
                            units = (
                                f" {item['units']}"
                                if "units" in item and item["units"]
                                else ""
                            )
                            combined_values.append(
                                f"{component_name}: {component_value}{units}"
                            )

                    combined_result["value"] = ", ".join(combined_values)

                    # Add any notes if present
                    notes = [
                        item.get("note")
                        for item in test_result
                        if "note" in item and item["note"]
                    ]
                    if notes:
                        combined_result["note"] = "; ".join(notes)

                    # Replace the list with the combined dictionary
                    data[category][test_name] = combined_result

        return data

    @property
    def prompt(self):
        _prompt = ""

        for k in Investigations.model_fields.keys():
            if self[k]:
                _prompt += prettify(k) + "\n"
                for ix_key, ix_v in self[k].items():
                    if isinstance(ix_v, InvestigationResult):
                        _prompt += ix_v.prompt
                    else:
                        _prompt += f"- {ix_key}: {prettify(ix_v)}"
                    _prompt += "\n"

        return _prompt if _prompt else "Nil significant investigations"

    @model_validator(mode="after")
    def format_specimen_type(self):

        for _specimen_type in Investigations.model_fields.keys():

            if self[_specimen_type]:
                _investigations = {}
                for k, v in self[_specimen_type].items():
                    if isinstance(v, InvestigationResult):
                        if not v.specimen_type:
                            v.specimen_type = _specimen_type
                        _investigations[k] = v

                self.__setattr__(_specimen_type, _investigations)

        return self

    def keys(self):
        keys = []
        for specimen_type in Investigations.model_fields.keys():
            if self[specimen_type]:
                keys.extend(
                    [f"{specimen_type}:{specimen}" for specimen in self[specimen_type]]
                )

        return keys

    def dict(self):
        items = {}
        for specimen_type in Investigations.model_fields.keys():
            if self[specimen_type]:
                for specimen in self[specimen_type]:
                    if isinstance(specimen, InvestigationResult):
                        items[f"{specimen_type}:{specimen}"] = specimen.prompt
                    else:
                        items[f"{specimen_type}:{specimen}"] = str(specimen)
        return items

    def get_prompt(self, query: str):
        if ":" in query:
            if query in self.keys():
                specimen_type, ix_key = query.split(":")
                ix_result = self[specimen_type][ix_key]

                return (
                    ix_result.prompt
                    if isinstance(ix_result, InvestigationResult)
                    else f"- {prettify(ix_key)}: {prettify(ix_result)}."
                )
            else:
                return self.get_default(query)
        else:
            return ""

    # TODO: Get normal reference values here later
    def get_default(self, query):
        specimen_type, ix_key = query.split(":")

        return f"- {prettify(ix_key)}: Normal"

    def get_grouped_investigations(self, queries: List[str]):

        ix_by_specimen = defaultdict(list)
        for query in queries:
            if len(query.split(":")) == 2:
                specimen_type, _ = query.split(":")
                if ix_prompt := self.get_prompt(query):
                    ix_by_specimen[specimen_type].append(ix_prompt)

        _prompt = ""

        for k, v in ix_by_specimen.items():
            _prompt += k.capitalize() + ":\n"
            _prompt += " \n".join(v) + "\n"

        return _prompt

    def dict(self):
        items = {}
        for specimen_type in Investigations.model_fields.keys():
            if self[specimen_type]:
                for ix_name, ix_value in self[specimen_type].items():
                    items[f"{specimen_type}:{ix_name}"] = str(ix_value["name"])

        return items


class ImagingResult(ClinicalData):

    modality: str
    region: str
    report: str

    @property
    def name(self):
        return f"{self.modality.upper()} {self.region.upper()}"

    @property
    def prompt(self):
        return f"== {self.modality.upper()} {self.region.upper()} ==\n{self.report}\n"


class Differential(ClinicalData):

    name: str
    icd_10: str
    icd_10_name: Optional[str] = None
    relevant_keys: List[str] = Field(default_factory=list)
    reasoning: str = ""


class ClinicalCase(ClinicalData):
    demographics: Demographics
    history: History
    history_freetext: Optional[str] = None
    physical: PhysicalExamination
    investigations: Investigations
    imaging: Dict[str, ImagingResult] = Field(default_factory=dict)
    diagnosis_freetext: Optional[str] = (
        None  # Diagnosis before further parsing / validation
    )
    diagnosis: Optional[List[Differential]] = Field(default_factory=list)
    # List of acceptable differentials for this case
    differentials: Optional[List[Differential]] = Field(default_factory=list)

    def imaging_keys(self):
        return list(self.imaging.keys()) if self.imaging else []

    def imaging_dict(self):
        return {k: v.prompt for k, v in self.imaging.items()} if self.imaging else {}

    def keys(self):
        hx_keys = self.history.keys()
        phys_keys = self.physical.keys()
        ix_keys = self.investigations.keys()
        img_keys = self.imaging_keys()

        _keys = []
        _keys.extend([f"history:{k}" for k in hx_keys])
        _keys.extend([f"physical:{k}" for k in phys_keys])
        _keys.extend([f"investigation:{k}" for k in ix_keys])
        _keys.extend([f"imaging:{k}" for k in img_keys])

        return _keys

    def dict(self):
        hx_dict = self.history.dict()
        phys_dict = self.physical.dict()
        ix_dict = self.investigations.dict()
        img_dict = self.imaging_dict()

        _items = {}
        _items.update({f"history:{k}": v for k, v in hx_dict.items()})
        _items.update({f"physical:{k}": v for k, v in phys_dict.items()})
        _items.update({f"investigation:{k}": v for k, v in ix_dict.items()})
        _items.update({f"imaging:{k}": v for k, v in img_dict.items()})

        return _items

    def format_ddx(self, differential: Differential):
        factors = {
            "# History": [],
            "# Physical Examination": [],
            "# Investigations": [],
            "# Imaging": [],
        }

        for k in differential.relevant_keys:
            action_key = k.split(":", 1)
            if len(action_key) != 2:
                continue

            action, key = action_key
            if action == "history":

                sx = self.history.dict().get(key)
                if sx:
                    factors["# History"].append("- " + sx)
            elif action == "physical":
                factors["# Physical Examination"].append(
                    f"- {self.physical.get_prompt(key)}"
                )
            elif action == "investigation":
                factors["# Investigations"].append(self.investigations.get_prompt(key))
            elif action == "imaging":
                imaging = self.imaging.get(key)
                if imaging:
                    factors["# Imaging"].append("- " + imaging.name)
        factor_str = ""

        for factor_k, factor_v in factors.items():
            if factor_v and isinstance(factor_v, Iterable):
                _factor_v = [f for f in factor_v if f]
                factor_str += factor_k + "\n" + "\n".join(_factor_v) + "\n"

        _prompt = f"### {differential.name}"
        if differential.icd_10:
            _prompt += f"\n## ICD-10 code: {differential.icd_10}"
        if factor_str:
            _prompt += f"\n## Factors contributing to diagnosis:\n{factor_str}"
        if differential.reasoning:
            _prompt += f"\n## Reasoning: {differential.reasoning}"
        return _prompt

    @property
    def full_information(self) -> str:
        """Returns full information for the case, including diagnosis, for clinician review"""

        return (
            FULL_CASE_TEMPLATE.format(
                demographics=self.demographics.prompt,
                cc=self.history.chief_complaint,
                sx=self.history.symptom_list,
                pmh=self.history.pmh_list,
                sh=self.history.social_history_list,
                fh=self.history.family_history_list,
                mh=self.history.medication_list,
                allergies=self.history.allergies_list,
                physical=self.physical.prompt,
                investigations=self.investigations.prompt,
                imaging=(
                    "\n".join([i.prompt for i in self.imaging.values()])
                    if self.imaging
                    else "None"
                ),
            )
            + """\n==== Diagnosis
{diagnosis}

==== Other Differentials
{differentials}
""".format(
                diagnosis="\n".join(self.format_ddx(ddx) for ddx in self.diagnosis),
                differentials="\n".join(
                    self.format_ddx(ddx) for ddx in self.differentials
                ),
            )
        )

    @property
    def full_information_no_ddx(self) -> str:
        """Returns full information for the case, for full information benchmark"""

        return FULL_CASE_TEMPLATE.format(
            demographics=self.demographics.prompt,
            cc=self.history.chief_complaint,
            sx=self.history.symptom_list,
            pmh=self.history.pmh_list,
            sh=self.history.social_history_list,
            fh=self.history.family_history_list,
            mh=self.history.medication_list,
            allergies=self.history.allergies_list,
            physical=self.physical.prompt,
            investigations=self.investigations.prompt,
            imaging=(
                "\n".join([i.prompt for i in self.imaging.values()])
                if self.imaging
                else "None"
            ),
        )
