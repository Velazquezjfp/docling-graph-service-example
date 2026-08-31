# docling-graph-service — process flow

One `POST /v1/process` (or `dgs process`) runs three phases. Phases 1–2 are local CPU work inside the
container; phase 3 (and the optional VLM/embedding steps in phase 2) call your OpenAI-compatible endpoints.
The percentages you see in the logs (`Phase 1 (skeleton) … Phase 2 (fill) …`) are the two LLM sub-steps of
phase 3.

```mermaid
flowchart TB

    subgraph IN["INPUT — POST /v1/process"]
        direction TB
        DOCIN["document<br/>name · format: pdf | docx | pptx | html | md · base64_content"]
        CFGIN["pipeline_config · optional, per request<br/>ocr_enabled · vlm_enabled · embedding_enabled · graph_enabled<br/>chunk_max_tokens · extraction_contract: dense | direct"]
        ONTIN["ontology_graph · optional, per request<br/>inline ontology document — overrides the default<br/>default: graph.default_ontology_path = mounted /ontology/ontology.yaml"]
    end

    subgraph PH1["PHASE 1 — PARSE & STRUCTURE · docling, local CPU"]
        direction TB
        CONV["DocumentConverter<br/>tools: docling-parse PDF backend · layout model docling-layout-heron<br/>EasyOCR de+en when ocr_enabled · TableFormer ACCURATE for table structure<br/>images_scale 3.0 → page + picture rasters · heading hierarchy"]
        DCACHE[("conversion cache<br/>key: sha256 of file + ocr flag<br/>volume: /var/cache/dgs")]
        CLEAN["structure clean-up · service code<br/>page furniture: prefix rule + repeated header/footer detector<br/>heading levels from numbering: 5.3.1 → level 3<br/>de-hyphenate wrapped table cells · infer table/figure captions"]
        CONV --> CLEAN
        CONV <--> DCACHE
    end

    subgraph PH2["PHASE 2 — ENRICH, CHUNK, EMBED"]
        direction TB
        VLM["picture description · optional: vlm_enabled<br/>remote: POST vlm.base_url /chat/completions with image_url<br/>writes PictureItem.meta.description · German prompt<br/>failure → degraded.vlm, request continues"]
        MD["Markdown export<br/>docling export_to_markdown · image placeholders<br/>descriptions inline as Description lines"]
        CHUNK["chunking · docling HybridChunker + service rules<br/>tokenizer: intfloat multilingual-e5-large XLM-R · budget ≤ 512 tokens<br/>tables as own chunks: Markdown, caption + header on every part<br/>own peer-merge for text · hard cap splits, never truncates<br/>per chunk: breadcrumb · pages · dom_paths · TOPLEFT bboxes · token_count"]
        EMB["embeddings · optional: embedding_enabled<br/>remote: POST embedding.base_url /v1/embeddings · batches of 64<br/>dim checked against embedding.dim · order preserved<br/>failure → degraded.embeddings, request continues"]
        VLM --> MD --> CHUNK --> EMB
    end

    subgraph PH3["PHASE 3 — KNOWLEDGE GRAPH · docling-graph + service materializer"]
        direction TB
        TPL["ontology → template compiler · service code<br/>26 entity classes with graph_id_fields · 31 scalar link components<br/>synthetic ExtractionRoot with one catalog list per class"]
        DG1["docling-graph Phase 1 · skeleton discovery<br/>internal chunker: tiktoken cl100k + all-MiniLM tokenizer<br/>~15 LLM batches: find entity instances + identities<br/>then coverage pass over zero-yield chunks"]
        DG2["docling-graph Phase 2 · fill<br/>~64 LLM jobs: one per catalog path batch<br/>fills attributes + relation links per entity<br/>remote: llm.base_url via litellm · temperature 0 · structured output"]
        GC["GraphConverter → NetworkX DiGraph<br/>node ids from identity values · provenance binder: pages + doc refs<br/>alias reconciliation · integrity checks"]
        MAT["service materializer<br/>resolve link targets: identity → label → aliases · Dr. titles stripped<br/>edges with type · polarity + negation triggers · qualifier · quote · typed properties<br/>normalize dates, decimal commas, enums · violations kept + flagged<br/>unresolved targets reported, never stub nodes<br/>failure of phase 3 → degraded.graph, request continues"]
        TPL --> DG1 --> DG2 --> GC --> MAT
    end

    subgraph OUT["OUTPUT — ProcessResponse JSON"]
        direction TB
        O1["document: name · format · sha256 · pages · tables · pictures"]
        O2["markdown: clean text, descriptions inline"]
        O3["chunks: text · body_text · heading_breadcrumb · kind · caption<br/>page_numbers · dom_paths · bboxes · token_count · embedding"]
        O4["graph: nodes with attributes + provenance · edges with polarity + quote<br/>meta: counts · unresolved_targets · conflicts · violations"]
        O5["degraded: vlm · embeddings · graph — plus errors, warnings, timings_s, versions, cached"]
    end

    RCACHE[("result cache<br/>key: sha256 + options + ontology hash<br/>identical request → answer in < 1 s")]

    IN --> PH1 --> PH2 --> PH3 --> OUT
    IN -.identical request.-> RCACHE -.cached response.-> OUT
    ONTIN -. compiled per request .-> TPL

    classDef remote fill:#fff3e0,stroke:#e65100,color:#000
    classDef cache fill:#e3f2fd,stroke:#1565c0,color:#000
    class VLM,EMB,DG1,DG2 remote
    class DCACHE,RCACHE cache
```

Orange = steps that call your remote OpenAI-compatible endpoints; blue = caches. Every optional step degrades
gracefully: the response always contains markdown + chunks if the conversion succeeded, with `degraded.*`
flags and `errors[]` explaining what was skipped.

**Ontology as input:** `ontology_graph` in the request body always wins; the mounted
`/ontology/ontology.yaml` (via `graph.default_ontology_path`) is only the fallback when the request omits it.
An invalid inline ontology is rejected with `422` and precise field errors before any processing starts.
Compiled templates are cached per ontology hash, so alternating ontologies costs nothing.
