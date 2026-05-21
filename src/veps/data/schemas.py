"""Pydantic schemas for VEPS input files.

CVEObservation describes one row of data/raw/cve_observations.csv (required).
CVEMention / CVEMentionsFile describe data/raw/cve_mentions.json (optional).
Validation is performed by the loaders in src/veps/data/exploit_training_builder.py.
"""
import datetime
from typing import Dict

from pydantic import BaseModel, Field, RootModel


class CVEObservation(BaseModel):
    """One row of cve_observations.csv."""

    cve: str = Field(description="CVE ID, e.g. CVE-2021-44228.")
    date: datetime.date = Field(description="Observation date (YYYY-MM-DD).")
    count: int = Field(description="Number of observations on that date.")


class CVEMention(RootModel[Dict[datetime.date, int]]):
    """A single CVE's mention timeline: date -> count of mentions on that date."""


class CVEMentionsFile(RootModel[Dict[str, CVEMention]]):
    """Top-level shape of cve_mentions.json: CVE ID -> CVEMention."""
