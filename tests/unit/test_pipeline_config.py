from pathlib import Path

from conftest import ONTOLOGY_PATH
from docling_graph import PipelineConfig
from docling_graph.llm_clients.config import LlmRuntimeOverrides

from docling_graph_service.graph import pipeline_config, pipeline_kwargs
from docling_graph_service.ontology import compile_template, load_ontology
from docling_graph_service.settings import GraphSettings, LLMSettings


def test_every_key_exists_in_pipeline_config():
    compiled = compile_template(load_ontology(ONTOLOGY_PATH))
    llm = LLMSettings(api_key="k", model="gemini-dev")
    kwargs = pipeline_kwargs(Path("/tmp/x.json"), compiled, llm, GraphSettings())
    unknown = set(kwargs) - set(PipelineConfig.model_fields)
    assert not unknown, f"PipelineConfig silently ignores unknown keys: {unknown}"
    overrides = LlmRuntimeOverrides.model_validate(kwargs["llm_overrides"])  # sub-models are extra=forbid
    assert overrides.connection.base_url == "http://localhost:4000/v1"
    assert overrides.generation.max_tokens == llm.max_output_tokens == overrides.max_output_tokens
    cfg = pipeline_config(Path("/tmp/x.json"), compiled, llm, GraphSettings(), contract="direct")
    assert cfg.model_override == "openai/gemini-dev" and cfg.provider_override == "openai"
    assert cfg.extraction_contract == "direct" and cfg.dense_dedupe == "off" and cfg.dump_to_disk is False
    assert cfg.template is compiled.root
