"""Pydantic models + loaders for config/brands.yaml and config/trends.yaml.

These are the only functions that should read the config files: everything
downstream works against validated, normalized Python objects, and any
malformed config raises immediately (fail loud) rather than propagating
silently into a query or a parquet file.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from src.config import BRANDS_YAML, TIERS, TRENDS_YAML


class ConfigError(Exception):
    """Raised when config/brands.yaml or config/trends.yaml is malformed."""


# --- brands.yaml ---


class Brand(BaseModel):
    name: str
    search_term: str


class _BrandOverride(BaseModel):
    name: str
    search_term: str


def _normalize_brand(entry: Union[str, dict]) -> Brand:
    if isinstance(entry, str):
        return Brand(name=entry, search_term=entry)
    override = _BrandOverride.model_validate(entry)
    return Brand(name=override.name, search_term=override.search_term)


class BrandsConfig(BaseModel):
    tiers: Dict[str, List[Brand]]

    @model_validator(mode="after")
    def _check_tiers(self) -> "BrandsConfig":
        expected = set(TIERS)
        actual = set(self.tiers.keys())
        if actual != expected:
            raise ValueError(
                f"brands.yaml tier keys {sorted(actual)} do not match expected {sorted(expected)}"
            )
        for tier, brands in self.tiers.items():
            if not brands:
                raise ValueError(f"brands.yaml tier '{tier}' has no brands")
        return self

    @classmethod
    def from_yaml(cls, path: Path = BRANDS_YAML) -> "BrandsConfig":
        try:
            raw = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"Could not read/parse {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} did not parse to a mapping of tier -> brands")
        try:
            tiers = {
                tier: [_normalize_brand(entry) for entry in entries]
                for tier, entries in raw.items()
            }
            return cls(tiers=tiers)
        except Exception as exc:
            raise ConfigError(f"{path} failed schema validation: {exc}") from exc


# --- trends.yaml ---


class TrendEvidence(BaseModel):
    brands: List[str]
    note: Optional[str] = None
    source: HttpUrl


class Trend(BaseModel):
    id: str
    label: str
    query_terms: List[str] = Field(min_length=1)
    context: Optional[List[str]] = None
    evidence: TrendEvidence

    @field_validator("query_terms")
    @classmethod
    def _non_empty_terms(cls, v: List[str]) -> List[str]:
        if any(not t.strip() for t in v):
            raise ValueError("query_terms contains an empty/blank term")
        return v


class TrendsConfig(BaseModel):
    trends: List[Trend]

    @model_validator(mode="after")
    def _unique_ids(self) -> "TrendsConfig":
        ids = [t.id for t in self.trends]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"trends.yaml has duplicate trend ids: {sorted(dupes)}")
        if not self.trends:
            raise ValueError("trends.yaml has no trends")
        return self

    @classmethod
    def from_yaml(cls, path: Path = TRENDS_YAML) -> "TrendsConfig":
        try:
            raw = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"Could not read/parse {path}: {exc}") from exc
        if not isinstance(raw, dict) or "trends" not in raw:
            raise ConfigError(f"{path} did not parse to a mapping with a top-level 'trends' key")
        try:
            return cls.model_validate(raw)
        except Exception as exc:
            raise ConfigError(f"{path} failed schema validation: {exc}") from exc
