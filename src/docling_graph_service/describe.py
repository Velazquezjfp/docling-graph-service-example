"""Picture description with a remote VLM, written into ``PictureItem.meta.description``.

Our own step rather than docling's ``PictureDescriptionApiOptions`` hook: VLM on/off does not need a
second converter, failures are explicit (the hook swallows them into empty text), and the same call
can later serve the ADR-0010 "confirm" job.
"""

from __future__ import annotations

import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from docling_core.types.doc import DescriptionMetaField, DoclingDocument, PictureItem, PictureMeta

from .llm_http import LLMClient
from .settings import VLMSettings

log = logging.getLogger(__name__)


@dataclass
class DescribeReport:
    described: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def _area_fraction(pic: PictureItem, doc: DoclingDocument) -> float:
    if not pic.prov:
        return 1.0
    prov = pic.prov[0]
    page = doc.pages.get(prov.page_no)
    if page is None or not page.size.width or not page.size.height:
        return 1.0
    return (prov.bbox.width * prov.bbox.height) / (page.size.width * page.size.height)


def _to_png_data_url(image, max_px: int) -> str:
    if max(image.size) > max_px:
        ratio = max_px / max(image.size)
        image = image.resize((int(image.width * ratio), int(image.height * ratio)))
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def describe_picture(client: LLMClient, cfg: VLMSettings, image, caption: str | None, *, max_px: int) -> str:
    prompt = cfg.prompt
    if caption:
        prompt += f"\n\nBildunterschrift im Dokument: {caption}"
    payload = {
        "model": cfg.model,
        "temperature": 0.0,
        "max_tokens": cfg.max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _to_png_data_url(image, max_px)}},
            ],
        }],
    }
    resp = client.post_json("/chat/completions", payload, timeout_s=cfg.timeout_s)
    choice = resp["choices"][0]
    content = choice["message"].get("content") or ""
    if isinstance(content, list):  # some backends return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    text = content.strip()
    if not text:
        raise RuntimeError("VLM returned an empty description")
    if choice.get("finish_reason") == "length":
        # reasoning models spend max_tokens on thinking first: the visible answer is cut
        raise RuntimeError(f"VLM output truncated at max_tokens={cfg.max_tokens} (finish_reason=length); "
                           "raise vlm.max_tokens")
    return text


def describe_pictures(doc: DoclingDocument, client: LLMClient, cfg: VLMSettings, *, captions: dict[str, str] | None = None,
                      max_px: int = 2000) -> DescribeReport:
    report = DescribeReport()
    jobs: list[tuple[PictureItem, object, str | None]] = []
    for pic in doc.pictures:
        image = pic.get_image(doc)
        if image is None or _area_fraction(pic, doc) < cfg.min_area_fraction:
            report.skipped += 1
            continue
        caption = (captions or {}).get(pic.self_ref) or pic.caption_text(doc) or None
        jobs.append((pic, image, caption))

    def run(job):
        pic, image, caption = job
        try:
            return pic, describe_picture(client, cfg, image, caption, max_px=max_px), None
        except Exception as exc:  # noqa: BLE001 - per-picture degradation
            return pic, None, f"{pic.self_ref}: {exc}"

    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
        for pic, text, err in pool.map(run, jobs):
            if err:
                report.errors.append(err)
                continue
            desc = DescriptionMetaField(text=text, created_by=cfg.model)
            if pic.meta is None:
                pic.meta = PictureMeta(description=desc)
            else:
                pic.meta.description = desc
            report.described += 1
    return report
