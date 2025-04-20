# Base dataclass for ontology

from dataclasses import dataclass
from typing import List, Literal, Dict, Union
from vivabench.ontology.defaults import (
    DEFAULT_INVESTIGATIONS,
    DEFAULT_INVESTIGATION_SETS,
)
import json


### Base classes
@dataclass
class BaseConcept:
    """Base class for any medical concept, with methods to output as dictionary for storage as .json or as prompt"""

    def as_dict(self):
        return self.__dict__

    def prompt(self):
        output_prompt = ""
        for k, v in self.__dict__.items():
            output_prompt += f'{str.upper(k.replace("_", " "))}: {v}\n'
        return output_prompt


@dataclass
class HistoryOfPresentingComplaint(BaseConcept):  # History of Presenting Complaint
    brief_history: str
    full_history: str

    @classmethod
    def from_dict(cls, class_dict):
        return cls(**class_dict)


@dataclass
class Differential(BaseConcept):
    name: str
    reason: str = None
    urgency: str = None
    type: str = None

    @classmethod
    def from_dict(cls, class_dict):
        return cls(**class_dict)

    def prompt(self, full_ddx=False):
        output_prompt = "%s\n" % self.name.capitalize()
        if full_ddx:
            for attr in [self.reason, self.urgency, self.type]:
                if attr is not None:
                    output_prompt += ">> %s\n" % attr
        return output_prompt


@dataclass
class Investigation(BaseConcept):
    full_name: str
    value: str
    unit: str
    type: Literal["Vitals", "Serology", "Radiology", "Pathology", "EKG", "Other"]
    # alias: List[str] = None

    @classmethod
    def get_default(cls, ix_name):
        default_value = DEFAULT_INVESTIGATIONS.get(ix_name, None)

        if default_value is not None:
            return cls(**default_value)
        else:
            return None

    @classmethod
    def from_dict(cls, class_dict):
        return cls(**class_dict)

    def prompt(self):
        return f'{self.full_name}: {self.value} {self.unit if self.unit else ""}\n'


@dataclass
class Management(BaseConcept):
    # TODO: Should consider splitting this into different kinds of management, each with its base class. But this will do for now
    name: str
    value: str = ""  # Dosage
    unit: str = ""
    route: str = ""
    frequency: str = ""
    duration: str = ""
    type: str = ""  # TODO: To convert to specific types later

    @classmethod
    def from_dict(cls, class_dict):
        return cls(**class_dict)

    def prompt(self):
        if self.duration:
            return f"{self.name} {self.value}{self.unit} {self.route} {self.frequency} for {self.duration}\n"
        else:
            return (
                f"{self.name} {self.value}{self.unit} {self.route} {self.frequency}\n"
            )


### Composite classes
@dataclass
class PastMedicalHistory(BaseConcept):
    medical_history: str = "No significant medical history"
    surgical_history: str = "No significant surgical history"
    medications: str = "Patient currenlty not taking any medications"
    allergies: str = "No known drug allergies"
    family_history: str = "Parents both healthy. No significant family history"
    social_history: str = "Works as an office worker. Non-smoker"

    @classmethod
    def from_dict(cls, class_dict):
        return cls(**class_dict)


@dataclass
class SystemsReview(BaseConcept):
    general: str = "Noncontributory"
    skin: str = "Noncontributory"
    heent: str = "Noncontributory"
    pulmonary: str = "Noncontributory"
    cardiovascular: str = "Noncontributory"
    gastrointestinal: str = "Noncontributory"
    genitourinary: str = "Noncontributory"
    musculoskeletal: str = "Noncontributory"
    neurologic: str = "Noncontributory"
    psychiatric: str = "Noncontributory"

    @classmethod
    def from_dict(cls, class_dict):
        return cls(**class_dict)


@dataclass
class InvestigationSet(BaseConcept):
    name: str
    investigations: Dict[str, Investigation]

    @classmethod
    def get_default(cls, ix_name):
        requested_ix = DEFAULT_INVESTIGATION_SETS.get(ix_name, None)

        if requested_ix is not None:
            default_investigations = {
                ix: Investigation.get_default(ix) for ix in requested_ix
            }
            return cls(name=ix_name, investigations=default_investigations)
        else:
            return None

    @classmethod
    def from_dict(cls, class_dict):
        cls_investigations = {
            ix_name: Investigation.from_dict(ix_dict)
            for ix_name, ix_dict in class_dict["investigations"].items()
        }

        return cls(name=class_dict["name"], investigations=cls_investigations)

    def prompt(self, requested_ix: List = None):
        output_prompt = f'{self.name.upper().replace("_", " ")}\n'
        if requested_ix:
            for ix in requested_ix:
                output_prompt += "- %s" % self.investigations[ix].prompt()
        else:
            for ix in self.investigations.values():
                output_prompt += "- %s" % ix.prompt()
        output_prompt += "\n"

        return output_prompt

    def as_dict(self):
        class_dict = {}
        class_dict["name"] = self.name
        investigations = {}
        for k, ix in self.investigations.items():
            investigations[k] = ix.as_dict()

        class_dict["investigations"] = investigations
        return class_dict


@dataclass
class ConceptList:
    values: List[BaseConcept]

    def as_dict(self):
        return [val.as_dict() for val in self.values]

    def prompt(self, **prompt_kwargs):
        output_prompt = ""
        for value in self.values:
            output_prompt += "- %s" % value.prompt(**prompt_kwargs)
        output_prompt += "\n"
        return output_prompt

    @classmethod
    def from_dict(cls, values: List):
        return cls(values=values)


@dataclass
class ManagementList(ConceptList):
    values: List[Management] = None

    @classmethod
    def from_dict(cls, mx_list: List):
        return super().from_dict(values=[Management.from_dict(mx) for mx in mx_list])


@dataclass
class DifferentialList(ConceptList):
    values: List[Differential] = None

    @classmethod
    def from_dict(cls, dx_list: List):
        return super().from_dict(values=[Differential.from_dict(dx) for dx in dx_list])


@dataclass
class PhysicalExamination(BaseConcept):
    vitals: InvestigationSet = InvestigationSet.get_default("vitals")

    general: str = "Appears well"
    skin: str = "Warm to touch. No rash or cyanosis."
    heent: str = "Moist mucous membrane."
    pulmonary: str = "Breath sounds are equal bilaterally with good air movement in all fields. No wheezing."
    cardiovascular: str = (
        "Regular rate and rhythm; no murmurs, rubs, or gallop. No peripheral edema."
    )
    gastrointestinal: str = "Bowel sounds normal. Abdomen soft, non-tender."
    genitourinary: str = "Unremarkable"
    musculoskeletal: str = "Unremarkable"
    neurologic: str = "Unremarkable"
    mental_status: str = "Alert and oriented"

    @classmethod
    def from_dict(cls, class_dict):
        vitals = class_dict.get("vitals")
        if vitals is not None:
            class_dict["vitals"] = InvestigationSet.from_dict(vitals)
        else:
            class_dict["vitals"] = InvestigationSet.get_default("vitals")
        return cls(**class_dict)

    def as_dict(self):
        class_dict = self.__dict__.copy()
        for k, v in self.__dict__.items():
            if type(v) != str:
                class_dict[k] = v.as_dict()

        return class_dict

    def prompt(self):
        output_prompt = ""
        for k, v in self.__dict__.items():
            if type(v) != str:
                output_prompt += v.prompt()
            else:
                output_prompt += f"{str.upper(k)}: {v}\n"
        return output_prompt


# Hacky way to parse investigations for now
def _parse_ix_from_dict(class_dict):
    if "investigations" in class_dict.keys():
        return InvestigationSet.from_dict(class_dict)
    else:
        return Investigation.from_dict(class_dict)


@dataclass
class ClinicalCase:
    hopc: HistoryOfPresentingComplaint
    systems: SystemsReview
    pmh: PastMedicalHistory
    examination: PhysicalExamination
    primary_ddx: DifferentialList
    ix: Dict[str, Union[Investigation, InvestigationSet]]
    secondary_ddx: DifferentialList
    ddx: DifferentialList
    management: ManagementList

    def as_dict(self):
        class_dict = {
            "hopc": self.hopc.as_dict(),
            "history": self.systems.as_dict(),
            "pmh": self.pmh.as_dict(),
            "examination": self.examination.as_dict(),
            "primary_ddx": self.primary_ddx.as_dict(),
            "ix": {k: v.as_dict() for k, v in self.ix.items()},
            "likely_ddx": self.secondary_ddx.as_dict(),
            "ddx": self.ddx.as_dict(),
            "management": self.management.as_dict(),
        }

        return class_dict

    def prompt(self):
        output_prompt = ""

        output_prompt += "== HISTORY OF PRESENTING COMPLAINT\n%s\n" % self.hopc.prompt()
        output_prompt += "== SYSTEMS REVIEW\n%s\n" % self.systems.prompt()
        output_prompt += "== PAST MEDICAL HISTORY\n%s\n" % self.pmh.prompt()
        output_prompt += "== EXAMINATION\n%s\n" % self.examination.prompt()
        output_prompt += "== ESSENTIAL DIAGNOSIS\n%s\n" % self.primary_ddx.prompt()

        output_prompt += "== INVESTIGATIONS\n"
        for ix in self.ix.values():
            output_prompt += ix.prompt()

        output_prompt += (
            "== DIFFERENTIAL DIAGNOSIS AFTER INVESTIGATIONS\n%s\n"
            % self.secondary_ddx.prompt()
        )
        output_prompt += "== DIAGNOSIS\n%s\n" % self.ddx.prompt()
        output_prompt += "== MANAGEMENT\n%s\n" % self.management.prompt()

        return output_prompt

    @classmethod
    def from_dict(cls, case_dict):
        # TODO: Make validation checks later. But this will do for now
        hopc = HistoryOfPresentingComplaint.from_dict(case_dict["hopc"])
        systems = SystemsReview.from_dict(case_dict["history"])
        pmh = PastMedicalHistory.from_dict(case_dict["pmh"])
        examination = PhysicalExamination.from_dict(case_dict["examination"])
        primary_ddx = DifferentialList.from_dict(case_dict["primary_ddx"])
        ix = {
            ix_name: _parse_ix_from_dict(ix_dict)
            for ix_name, ix_dict in case_dict["ix"].items()
        }
        secondary_ddx = DifferentialList.from_dict(case_dict["likely_ddx"])
        ddx = DifferentialList.from_dict(case_dict["ddx"])
        management = ManagementList.from_dict(case_dict["management"])

        return cls(
            hopc=hopc,
            systems=systems,
            pmh=pmh,
            examination=examination,
            primary_ddx=primary_ddx,
            ix=ix,
            secondary_ddx=secondary_ddx,
            ddx=ddx,
            management=management,
        )
