### Hacky way to sort out files for now, will refactor later
DEFAULT_INVESTIGATIONS = {
    "temperature": {
        "full_name": "Temperature",
        "value": "36.5",
        "unit": "\u00b0C",
        "type": "vitals",
    },
    "pulse": {
        "full_name": "Pulse",
        "value": "60",
        "unit": "beats/minute",
        "type": "vitals",
    },
    "blood_pressure": {
        "full_name": "Blood Pressure",
        "value": "110/70",
        "unit": "mm Hg",
        "type": "vitals",
    },
    "respiratory_rate": {
        "full_name": "Respiratory Rate",
        "value": "12",
        "unit": "/ minute",
        "type": "vitals",
    },
    "sodium": {
        "full_name": "Sodium",
        "value": "140",
        "unit": "mmol/L",
        "type": "serology",
    },
}

DEFAULT_INVESTIGATION_SETS = {
    "vitals": ["temperature", "pulse", "blood_pressure", "respiratory_rate"]
}
