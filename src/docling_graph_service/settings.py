"""Configuration: ``config.yaml`` defaults, overridable per key via ``DGS__SECTION__KEY`` env vars.

Precedence (highest first): environment, ``.env`` file, ``config.yaml``, model defaults.
The YAML path comes from ``DGS_CONFIG`` (default: ``config.yaml`` in the working directory,
falling back to the copy shipped next to this package).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

_PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PACKAGE_DIR.parent.parent / "config.yaml"


class LLMSettings(BaseModel):
    """Extraction LLM behind the OpenAI-compatible proxy (drives docling-graph)."""

    base_url: str = "http://localhost:4000/v1"
    api_key: str = ""
    model: str = "gemini-dev"
    structured_output: bool = True
    temperature: float = 0.0
    timeout_s: float = 300.0
    max_retries: int = 2
    context_limit: int = 128_000
    max_output_tokens: int = 8192
    parallel_workers: int = 2


class VLMSettings(BaseModel):
    """Picture description via ``/chat/completions`` with image input."""

    enabled: bool = True
    base_url: str | None = None  # None -> llm.base_url
    api_key: str | None = None  # None -> llm.api_key
    model: str = "gemini-dev"
    prompt: str = (
        "Du siehst eine Abbildung aus einem deutschen IT-Betriebshandbuch. "
        "Beschreibe das technische Diagramm in zwei bis vier Sätzen auf Deutsch: "
        "nenne die dargestellten Systeme, Komponenten, Zonen und die Bedeutung der "
        "Pfeile und Beschriftungen. Gib nur wieder, was lesbar ist; erfinde nichts."
    )
    timeout_s: float = 120.0
    concurrency: int = 2
    max_tokens: int = 2000  # reasoning models count thinking tokens here; 400 truncates gemini-3.x
    min_area_fraction: float = 0.0


class EmbeddingSettings(BaseModel):
    """Batched ``/v1/embeddings``."""

    enabled: bool = True
    base_url: str | None = None
    api_key: str | None = None
    model: str = "bge-m3"
    dim: int = 1024
    batch_size: int = 64
    text_prefix: str = ""  # "passage: " for e5 models
    timeout_s: float = 120.0


class ChunkingSettings(BaseModel):
    tokenizer: str = "intfloat/multilingual-e5-large"
    max_tokens: int = 512
    merge_peers: bool = True  # merge adjacent text chunks under the same heading; tables/pictures stay separate
    strip_line_prefixes: list[str] = Field(default_factory=lambda: ["TESTDOKUMENT ·"])
    repeated_furniture_min_pages: int = 3  # 0 disables the repeated header/footer detector
    furniture_band: float = 0.12  # top/bottom fraction of the page where runners live


class DoclingSettings(BaseModel):
    ocr: bool = True
    ocr_langs: list[str] = Field(default_factory=lambda: ["de", "en"])
    images_scale: float = 3.0
    table_mode: Literal["accurate", "fast"] = "accurate"
    heading_hierarchy: bool = True
    numbered_heading_levels: bool = True  # "5.3.1 …" -> level 3; unnumbered headings -> parent + 1
    dehyphenate_table_cells: bool = True  # "bavd- issuing-ca-3" -> "bavd-issuing-ca-3" (PDF line wraps)
    num_threads: int = 4
    artifacts_path: str | None = None  # None -> DOCLING_ARTIFACTS_PATH / online download
    document_timeout_s: float | None = 900.0


class GraphSettings(BaseModel):
    enabled: bool = True
    extraction_contract: Literal["dense", "direct"] = "dense"
    dense_dedupe: Literal["off", "standard"] = "off"
    provenance: Literal["standard", "off"] = "standard"
    chunk_max_tokens: int = 512  # docling-graph's *internal* chunker, not the embedding chunks
    default_ontology_path: str | None = None


class ServiceSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    max_upload_mb: int = 50
    request_deadline_s: float = 1800.0
    cache_dir: str | None = None
    log_level: str = "INFO"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DGS__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    vlm: VLMSettings = Field(default_factory=VLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    docling: DoclingSettings = Field(default_factory=DoclingSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = config_path()
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        if yaml_path is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        return tuple(sources)

    @model_validator(mode="after")
    def _inherit_endpoint_defaults(self) -> Settings:
        for section in (self.vlm, self.embedding):
            if section.base_url is None:
                section.base_url = self.llm.base_url
            if section.api_key is None:
                section.api_key = self.llm.api_key
        return self

    @property
    def artifacts_path(self) -> str | None:
        return self.docling.artifacts_path or os.environ.get("DOCLING_ARTIFACTS_PATH") or None


def config_path() -> Path | None:
    """Resolve the YAML config file: ``DGS_CONFIG`` > ``./config.yaml`` > packaged default."""
    explicit = os.environ.get("DGS_CONFIG")
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"DGS_CONFIG points to a missing file: {p}")
        return p
    for candidate in (Path.cwd() / "config.yaml", _DEFAULT_CONFIG):
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def redact(url: str, key: str | None) -> dict[str, str]:
    """Endpoint description safe for ``/v1/capabilities``."""
    return {"base_url": url, "api_key": "***" if key else "(none)"}
