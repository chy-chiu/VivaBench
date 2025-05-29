import random
import re

FULL_CASE_TEMPLATE = """{demographics} presenting with {cc}. 
==== Symptoms
{sx}
==== Past Medical History
{pmh}
Allergies: {allergies}
==== Social History
{sh}
==== Family History
{fh}
==== Medications
{mh}

==== Examination
{physical}

==== Investigations
{investigations}

==== Imaging
{imaging}
"""


# Symptom templates
def get_timing_phrases(phrase: str):
    timing_phrases = {
        "acute": [
            "happened very suddenly",
            "started all of a sudden",
            "came on abruptly",
            "began without warning",
            "occurred out of nowhere",
        ],
        "subacute": [
            "happened yesterday",
            "started a couple of days ago",
            "began recently",
            "developed over the past few days",
            "has been going on for a short while",
        ],
        "gradual": [
            "happened gradually",
            "developed slowly over time",
            "came on little by little",
            "progressed over several weeks",
            "worsened slowly",
        ],
        "chronic": [
            "has been going on forever",
            "has persisted for a long time",
            "has been present for years",
            "has lasted for ages",
            "has been ongoing for as long as I can remember",
        ],
        "recurrent": [
            "comes and goes",
            "happens off and on",
            "recurs from time to time",
            "keeps coming back",
            "occurs intermittently",
        ],
        "spontaneous": [
            "was quite spontaneous",
            "happened without any clear reason",
            "occurred unexpectedly",
            "came out of the blue",
            "started for no apparent reason",
        ],
    }

    ESCAPE_CLAUSE = "has been going on for a bit, but I am not sure when it happened."

    if phrase in timing_phrases:
        return random.choice(timing_phrases[phrase])
    else:
        return ESCAPE_CLAUSE


timing_adjective_map = {
    "abrupt": "acute",
    "acute": "acute",
    "immediate": "acute",
    "sudden": "acute",
    "subacute": "subacute",
    "gradual": "gradual",
    "insidious": "gradual",
    "chronic": "chronic",
    "recent": "acute",
    "episodic": "recurrent",
    "intermittent": "recurrent",
    "recurrent": "recurrent",
    "spontaneous": "spontaneous",
    "resolved": "spontaneous",
}

header_a_keys = ["on", "since", "after"]
header_b_keys = ["prior", "at birth", "new", "congenital", "secondary", "yesterday"]
verb_keys = ["preced"]


def phrase_onset(phrase):
    header_a = "happened "  # For -ly phrases
    header_b = "was "  # For adjective phrases

    if any(phrase.lower().startswith(k) for k in header_a_keys):
        return header_a + phrase

    # If it has brackets, convert to "- "
    if "(" in phrase and ")" in phrase:
        # e.g., "gradual (started yesterday)" -> "gradually - started yesterday"
        main, extra = re.match(r"^(.*?)\s*\((.*?)\)$", phrase).groups()
        #  Modify timing phrases if any
        if main in timing_adjective_map.keys():
            main = get_timing_phrases(timing_adjective_map[main])
        return f"{main} - {extra}"

    # In general screen for timing adjectives
    if any(phrase.lower().startswith(k) for k in timing_adjective_map.keys()):
        return get_timing_phrases(timing_adjective_map[phrase.split(" ")[0]])

    if any(phrase.lower().startswith(k) for k in header_b_keys):
        return header_b + phrase

    return f"{phrase}. "


time_descriptive = ["second", "minute", "hour", "day", "week", "month", "year"]
frequency_words = [
    "few",
    "several",
    "multiple",
    "couple",
    "times",
    "each",
    "one",
    "two",
]
gestation_keywords = ["gestation", "trimester", "week of life", "pregnant"]
non_duration_keywords = [
    "acute",
    "chronic",
    "progressive",
    "resolved",
    "intermittent",
    "persistent",
    "recent",
    "prolonged",
    "self-limited",
    "recurrent",
    "multiple episodes",
    "not specified",
    "unspecified",
    "ongoing",
    "lifelong",
    "long-standing",
]


def phrase_duration(phrase):
    p = phrase.lower().strip()

    # Filter out non-duration
    if any(kw in p for kw in non_duration_keywords):
        return f"was {p} in duration"

    # Gestational age
    if "gestation" in p:
        match = re.search(r"(\d+)\s*weeks? gestation", p)
        if match:
            return f"{match.group(1)} weeks pregnant"
        else:
            return p
    if "trimester" in p:
        return f"in the {p}"
    if "week of life" in p:
        return p.replace("week of life", "week of life (neonate)")
    if "pregnant" in p:
        return p

    # Since phrases
    if p.startswith("since"):
        return f"since {phrase[6:]}"

    # Over the past, past, last
    if p.startswith("over the past"):
        return f"over the past {phrase[14:]}"
    if p.startswith("past "):
        return f"over the past {phrase[5:]}"
    if p.startswith("last "):
        return f"over the past {phrase[5:]}"

    # Direct durations
    if any(unit in p for unit in time_descriptive):
        return f"happened {phrase}" if "ago" in p else f"going on for {phrase}"

    # Frequency
    if any(word in p for word in frequency_words):
        return f"happens {phrase}"

    # "Episode" or "episodes"
    if "episode" in p:
        return f"happened for {phrase}"

    # "Earlier today", "today", "yesterday"
    if "today" in p or "yesterday" in p:
        return f"since {phrase}"

    # "About", "almost", "within"
    if p.startswith("about ") or p.startswith("almost ") or p.startswith("within "):
        return f"going on for {phrase}"

    # "During"
    if p.startswith("during "):
        return p

    # Default: return as is
    return f"had {phrase} in duration"


def _symptom_description(symptom, addit_keys=[]):

    symptom_ref = symptom.name.capitalize()

    prompt_parts = []

    if "severity" in addit_keys:
        if symptom.severity:
            prompt_parts.append(f"The {symptom_ref.lower()} was {symptom.severity}.")
        else:
            prompt_parts.append(
                f"The {symptom_ref.lower()} was hard to describe in terms of severity."
            )

    # Refers as 'It' after to make it sound natural
    if "onset" in addit_keys:
        if symptom.onset:
            prompt_parts.append(
                f"{symptom_ref if not prompt_parts else 'It'} {phrase_onset(symptom.onset)}."
            )
        else:
            prompt_parts.append(
                f"{symptom_ref if not prompt_parts else 'It'} had unsure onset."
            )

    if "duration" in addit_keys:
        if symptom.duration:
            prompt_parts.append(
                f"{symptom_ref if not prompt_parts else 'It'} {phrase_duration(symptom.duration)}."
            )
        else:
            prompt_parts.append(
                f"{symptom_ref if not prompt_parts else 'It'} had unsure duration."
            )

    if "character" in addit_keys:
        if symptom.character:
            prompt_parts.append(
                f"{symptom_ref + ' -' if not prompt_parts else 'It feels'} {symptom.character}."
            )
        else:
            prompt_parts.append(
                f"{symptom_ref + ' -' if not prompt_parts else 'It'} 'just felt abnormal' as per the patient."
            )

    if "location" in addit_keys:
        if symptom.location:
            prompt_parts.append(
                f"{symptom_ref if not prompt_parts else 'It'} happens around {symptom.location}."
            )
        else:
            prompt_parts.append(f"Patient was unable to pinpoint location.")

    if "radiation" in addit_keys:
        if symptom.radiation:
            prompt_parts.append(
                f"Sometimes {'the ' + symptom_ref.lower() if not prompt_parts else 'It'} spreads to {symptom.radiation}."
            )
        else:
            prompt_parts.append(f"{symptom_ref} doesn't spread anywhere else")

    if "alleviating" in addit_keys:
        if symptom.alleviating_factors:
            factors = ", ".join([s.lower() for s in symptom.alleviating_factors])
            prompt_parts.append(
                f"{symptom_ref if not prompt_parts else 'It'} seems to get better with {factors}."
            )
        else:
            sx = "the" + symptom_ref.lower()
            prompt_parts.append(
                f"Nothing makes {sx if not prompt_parts else 'it'} better."
            )

    if "aggravating" in addit_keys:

        if symptom.aggravating_factors and ("aggravating" in addit_keys):
            factors = ", ".join([s.lower() for s in symptom.aggravating_factors])
            prompt_parts.append(
                f"{symptom_ref if not prompt_parts else 'It'} tends to worsen when I {factors}."
            )
        else:
            sx = "the" + symptom_ref.lower()

            prompt_parts.append(
                f"Nothing makes {sx if not prompt_parts else 'it'} worse."
            )

    if "associated" in addit_keys:
        if symptom.associated_symptoms:
            symptoms = ", ".join([s.lower() for s in symptom.associated_symptoms])
            prompt_parts.append(
                f"Along with this, the patient also noticed {symptoms}."
            )
        else:
            prompt_parts.append(f"No other associated symptoms.")

    if "context" in addit_keys:
        if symptom.context:
            prompt_parts.append(
                f"Other relevant context for {symptom_ref.lower()}: {symptom.context}"
            )
        elif symptom.notes:
            prompt_parts.append(f"Note for {symptom_ref.lower()}: {symptom.notes}")

    return " ".join(prompt_parts)
