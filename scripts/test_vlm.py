#!/usr/bin/env python3
"""Standalone VLM endpoint tester — finds out whether (and via which request format) a model
actually SEES images, instead of politely answering without them.

Sends a real ZSD diagram (tests/data/zsd1-uebersicht-small.png) and grades the reply
deterministically: the image contains tokens no blind model can guess (hostnames like
`ca-root-off01`, the domain `bavd.intern`, IPs `10.20.6.x`). No LLM judge needed.

Probes, in order:
  show            Ollama /api/show     -> does the model tag declare the "vision" capability?
  openai-object   /v1/chat/completions -> content part {"type":"image_url","image_url":{"url":"data:..."}}
                                          (EXACTLY what the service's describe_picture() sends)
  openai-string   /v1/chat/completions -> "image_url" as a bare data-URL string (some gateways)
  native-chat     Ollama /api/chat     -> messages[0].images = [<raw base64, no data: prefix>]
  native-generate Ollama /api/generate -> prompt + images = [<raw base64>]
  native-chat-pre same as native-chat, but with an OCR-preprocessed image (white background,
                  patch-snapped LANCZOS resize, autocontrast — deliberately NO sharpening,
                  which measurably hurt gemma4) — A/B against native-chat to see whether
                  preprocessing lifts reading fidelity. Needs Pillow; skipped otherwise.

Usage (remote server, against the tunnel):
    python3 test_vlm.py --url http://localhost:11435 --model gemma4:latest
Usage (local, against Ollama directly):
    python3 test_vlm.py --url http://localhost:11434 --model gemma4:e2b
Against a proxy like LiteLLM (native probes are skipped automatically on 404):
    python3 test_vlm.py --url http://localhost:4000 --model gemini-dev --api-key sk-...

stdlib only, Python 3.8+. Ignores every proxy environment variable on purpose.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# never let a corporate proxy intercept explicit localhost/tunnel URLs
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

DEFAULT_IMAGE = Path(__file__).resolve().parents[1] / "tests" / "data" / "zsd1-uebersicht-small.png"

PROMPT = (
    "Du siehst ein technisches Diagramm. Liste wörtlich auf, was im Bild lesbar ist:\n"
    "1. die Titel aller Kästen/Boxen,\n"
    "2. drei sichtbare Hostnamen oder Domains,\n"
    "3. zwei sichtbare IP-Adressen oder Portnummern.\n"
    "Nur auflisten, was tatsächlich im Bild steht — nichts erfinden."
)

# Tokens printed in the diagram, in three tiers:
#   SPECIFIC — fine print (hostnames/domains/IDs). Reading these = production-grade vision.
#   TITLE   — large box titles/headline. Unguessable, so any hit proves the image ARRIVED,
#             even when the model's vision resolution is too low for the fine print.
#   GENERIC — guessable architecture words; weak signal, reported only.
SPECIFIC_TOKENS = ["bavd.intern", "ca-root-off01", "hsm-p01", "jump01", "10.20.6",
                   "vault-system", "BHB-PLT-0007", "es-grafana", "dd-gateway",
                   "vpp-portal", "obs-vpp", "pg-iam-p01"]
TITLE_TOKENS = ["Zentrale Sicherheitsdienste", "Konsumenten", "Dokumentendienste", "Checkmk",
                "OpenSearch", "BAVD", "Keycloak", "Event-System", "ocp-prod",
                "Administration ZSD", "OpenShift-Konsole"]
GENERIC_LABELS = ["Keycloak", "Vault", "Kafka", "Grafana", "VPP", "PKI", "IAM",
                  "OpenShift", "Dokumentendienste", "Active Directory", "CaaS"]
BLIND_PHRASES = ["provide the image", "provide an image", "no image", "cannot see", "can't see",
                 "don't see an image", "unable to view", "kein bild", "sehe kein",
                 "stellen sie das bild", "bild bereitstellen", "bild zur verfügung"]


def http_json(url: str, payload: dict | None, api_key: str | None, timeout: float):
    """POST payload as JSON (or GET when payload is None). Returns (status, parsed-or-text, seconds)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    t0 = time.time()
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        status = exc.code
    dt = time.time() - t0
    try:
        return status, json.loads(body), dt
    except ValueError:
        return status, body, dt


def load_image(path: Path) -> tuple[str, str]:
    """Returns (raw_base64, data_url) — the data URL matches describe._to_png_data_url()."""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return b64, f"data:{mime};base64,{b64}"


def preprocess_image(path: Path, max_dim: int = 1540, patch: int = 14) -> tuple[str, str] | None:
    """OCR-friendly preprocessing for ViT-style vision encoders. Returns (base64, note), or
    None when Pillow is missing. The result is also saved as <image>-pre.png for eyeballing."""
    try:
        from PIL import Image, ImageOps, ImageStat
    except ImportError:
        return None
    img = Image.open(path)
    # flatten transparency onto white — encoders render alpha as black, drowning dark text
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    img = img.convert("RGB")
    # invert dark-mode diagrams: encoders and OCR are trained mostly on dark-on-light
    if ImageStat.Stat(img.convert("L")).mean[0] < 80:
        img = ImageOps.invert(img)
    # high-quality LANCZOS downscale snapped to the encoder's patch grid — better to do the
    # resize ourselves than let the server do it with a cheap filter
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    tw = max(patch, round(w * scale / patch) * patch)
    th = max(patch, round(h * scale / patch) * patch)
    if (tw, th) != (w, h):
        img = img.resize((tw, th), Image.Resampling.LANCZOS)
    # adaptive contrast stretch. NO sharpening: an unsharp mask measurably DEGRADED gemma4's
    # reading (halos read as noise after the encoder's own downsampling) — bisected 2026-09-01.
    img = ImageOps.autocontrast(img, cutoff=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    out = path.with_name(path.stem + "-pre.png")
    try:
        out.write_bytes(buf.getvalue())
        saved = f"saved as {out.name}"
    except OSError:
        saved = "not saved (directory read-only)"
    note = f"{w}x{h} -> {img.width}x{img.height}, patch-snapped LANCZOS + autocontrast, {saved}"
    return base64.b64encode(buf.getvalue()).decode("ascii"), note


def count_hits(text: str) -> tuple[int, int, int]:
    low = text.casefold()
    return (sum(1 for t in SPECIFIC_TOKENS if t.casefold() in low),
            sum(1 for t in TITLE_TOKENS if t.casefold() in low),
            sum(1 for t in GENERIC_LABELS if t.casefold() in low))


def grade(reply: str, reasoning: str = "") -> tuple[str, tuple[int, int, int]]:
    if not reply.strip():
        # visible answer truncated (thinking models) — the hidden reasoning may still prove vision
        hits = count_hits(reasoning)
        specific, title, _ = hits
        if specific >= 2:
            return "PASS*", hits  # sees the fine print, but the visible answer was cut off
        if title >= 2:
            return "LOWRES*", hits
        return "TRUNCATED", hits
    hits = count_hits(reply)
    specific, title, generic = hits
    if specific >= 2:
        return "PASS", hits
    if any(p in reply.casefold() for p in BLIND_PHRASES) and title == 0:
        return "BLIND", hits
    if title >= 2:
        return "LOWRES", hits  # image arrives, but the model cannot read the fine print
    if specific == 0 and title <= 1 and generic <= 2:
        return "BLIND", hits
    return "UNCLEAR", hits


def flatten_openai_content(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, list):  # some backends return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content.strip()


def show_reply(reply: str, limit: int = 700) -> str:
    reply = " ".join(reply.split())
    return reply[:limit] + ("…" if len(reply) > limit else "")


def probe_show(base: str, model: str, api_key: str | None, timeout: float) -> list | None:
    """Returns the model's capability list (or [] derived from tensors), None when /api/show is unavailable."""
    status, body, _ = http_json(f"{base}/api/show", {"model": model}, api_key, timeout)
    if status == 404 or not isinstance(body, dict) or "error" in body and status >= 400:
        print(f"  /api/show not available (HTTP {status}) — not an Ollama endpoint? "
              f"Skipping the capability check.")
        if isinstance(body, dict) and body.get("error"):
            print(f"  server said: {str(body['error'])[:200]}")
        return None
    caps = body.get("capabilities")
    info = body.get("model_info", {})
    arch = info.get("general.architecture", "?")
    ctx = info.get(f"{arch}.context_length")
    vision_keys = [k for k in info if ".vision." in k]
    print(f"  architecture={arch}  context_length={ctx}  capabilities={caps}")
    if caps is None:
        print(f"  (old Ollama: no capabilities field; vision tensors present: {bool(vision_keys)})")
        return ["vision"] if vision_keys else []
    return caps


def build_variants(model: str, prompt: str, b64: str, data_url: str, max_tokens: int,
                   thinking: bool, b64_pre: str | None = None) -> dict:
    # think:false only where the API supports it (native) and the model declares thinking —
    # otherwise a thinking model burns the whole token budget on hidden reasoning.
    think = {"think": False} if thinking else {}
    variants = {
        "openai-object": ("/v1/chat/completions", {
            "model": model, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        }),
        "openai-string": ("/v1/chat/completions", {
            "model": model, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": data_url},
            ]}],
        }),
        "native-chat": ("/api/chat", {
            "model": model, "stream": False, **think,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }),
        "native-generate": ("/api/generate", {
            "model": model, "stream": False, **think, "prompt": prompt, "images": [b64],
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }),
    }
    if b64_pre is not None:
        # Same call as native-chat, but with the OCR-preprocessed image — A/B for fidelity.
        # The prompt gets a distinct first line: Ollama's prompt-prefix cache matches the text
        # tokens of the previous same-prompt call and then reuses stale image state for the NEW
        # image (empty EOS replies, reproduced on gemma4 AND qwen3.5) — a token-distinct prefix
        # forces a clean prefill.
        variants["native-chat-pre"] = ("/api/chat", {
            "model": model, "stream": False, **think,
            "messages": [{"role": "user", "content": "Neues Bild, neue Aufgabe.\n" + prompt,
                          "images": [b64_pre]}],
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        })
    return variants


def extract_reply(name: str, body) -> tuple[str, str, str]:
    """Returns (reply_text, hidden_reasoning_text, finish_note)."""
    if not isinstance(body, dict):
        return "", "", ""
    if name.startswith("openai"):
        choices = body.get("choices") or []
        if not choices:
            return "", "", ""
        msg = choices[0].get("message", {})
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        finish = choices[0].get("finish_reason", "")
        return flatten_openai_content(msg), reasoning, (f"finish_reason={finish}" if finish else "")
    if name == "native-chat":
        msg = body.get("message", {})
        note = "done_reason=" + str(body.get("done_reason", ""))
        return (msg.get("content") or "").strip(), msg.get("thinking") or "", note
    note = "done_reason=" + str(body.get("done_reason", ""))
    return (body.get("response") or "").strip(), body.get("thinking") or "", note


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:11434",
                    help="endpoint base, WITHOUT /v1 (e.g. http://localhost:11435 for the tunnel)")
    ap.add_argument("--model", required=True, help="model name/tag, e.g. gemma4:latest")
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--api-key", default=os.environ.get("DGS__VLM__API_KEY") or None)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-tokens", type=int, default=1200,
                    help="generous default: thinking models spend tokens before answering")
    ap.add_argument("--variants",
                    default="openai-object,openai-string,native-chat,native-generate,native-chat-pre")
    ap.add_argument("--pre-max-dim", type=int, default=1540,
                    help="max dimension for the preprocessed image (native-chat-pre variant)")
    ap.add_argument("--pre-patch", type=int, default=14,
                    help="ViT patch grid the resize snaps to: 14 for gemma-style encoders, "
                         "28 for the qwen-VL family (14px patches merged 2x2)")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    if not args.image.is_file():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2
    b64, data_url = load_image(args.image)
    print(f"endpoint: {base}   model: {args.model}")
    print(f"image: {args.image.name} ({len(b64) * 3 // 4 // 1024} KiB, {len(b64) // 1024} KiB as base64)")

    print("\n== probe: /api/show (vision capability)")
    caps = probe_show(base, args.model, args.api_key, min(args.timeout, 30))
    vision = None if caps is None else "vision" in caps
    thinking = bool(caps) and "thinking" in caps
    if vision is True:
        print("  -> model tag declares VISION. Images should work; testing request formats.")
        if thinking:
            print("  -> model also THINKS: native calls get think:false; OpenAI-compat cannot"
                  " disable it, so hidden reasoning is graded as vision evidence too.")
    elif vision is False:
        print("  -> model tag has NO vision capability. No request format can fix this —")
        print("     Ollama silently drops the images. Pull a vision-capable tag of the model.")

    pre = preprocess_image(args.image, args.pre_max_dim, args.pre_patch)
    if pre is None:
        print("\n(Pillow not installed — skipping the native-chat-pre variant."
              " `pip install pillow` to enable it.)")
        b64_pre = None
    else:
        b64_pre, pre_note = pre
        print(f"\npreprocessed image: {pre_note} ({len(b64_pre) // 1024} KiB as base64)")

    results: dict[str, tuple[str, str]] = {}
    variants = build_variants(args.model, PROMPT, b64, data_url, args.max_tokens, thinking, b64_pre)
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if name == "native-chat-pre" and b64_pre is None:
            continue  # Pillow missing, already announced
        if name not in variants:
            print(f"unknown variant: {name}", file=sys.stderr)
            continue
        path, payload = variants[name]
        print(f"\n== variant: {name}  POST {base}{path}")
        # breathe between requests: firing within milliseconds of the previous response
        # reproduces an Ollama runner race that returns empty multimodal replies
        time.sleep(3.0)
        for attempt in (1, 2):
            status, body, dt = http_json(f"{base}{path}", payload, args.api_key, args.timeout)
            if status >= 400:
                break
            reply, reasoning, note = extract_reply(name, body)
            if reply.strip() or reasoning or "length" in note or attempt == 2:
                break
            # Empty answer without truncation: observed as an Ollama runner bug — the byte-
            # identical request (verified via a logging relay, same body sha) fails when its
            # multimodal prefill directly follows /api/show or a model (re)load, and succeeds
            # after any completed chat. So: complete a tiny text-only chat, then retry.
            print(f"  empty reply ({note}) after {dt:.1f}s — warmup chat, then retrying once")
            try:
                http_json(f"{base}/api/chat", {
                    "model": args.model, "stream": False,
                    "messages": [{"role": "user", "content": "Sag OK."}],
                    "options": {"num_predict": 10},
                }, args.api_key, 120)
            except Exception:  # noqa: BLE001 - warmup is best-effort
                pass
        if status >= 400:
            err = body.get("error") if isinstance(body, dict) else body
            print(f"  HTTP {status} after {dt:.1f}s: {str(err)[:300]}")
            results[name] = ("ERROR", f"HTTP {status}")
            continue
        verdict, (specific, title, generic) = grade(reply, reasoning)
        if verdict == "TRUNCATED" and "length" not in note:
            verdict = "EMPTY"
        results[name] = (verdict, f"{specific} fine-print / {title} title / {generic} generic")
        print(f"  HTTP {status} in {dt:.1f}s  {note}")
        print(f"  reply: {show_reply(reply) or '(empty)'}")
        if not reply.strip() and reasoning:
            print(f"  hidden reasoning: {show_reply(reasoning, 400)}")
        print(f"  -> {verdict}  (fine-print: {specific}/{len(SPECIFIC_TOKENS)}, "
              f"titles: {title}/{len(TITLE_TOKENS)}, generic: {generic}/{len(GENERIC_LABELS)})")
        if verdict.startswith("LOWRES"):
            print("     the image ARRIVES (title text read correctly), but the model cannot read")
            print("     the fine print — a vision-quality problem, not a transport problem.")
        if verdict == "PASS*":
            print("     sees the image, but the visible answer was truncated by max_tokens —")
            print("     raise --max-tokens, or disable thinking for this model in production.")
        elif verdict == "TRUNCATED":
            print("     empty answer, finish=length: no room for a visible reply — retry with"
                  " a higher --max-tokens before drawing conclusions.")
        elif verdict == "EMPTY":
            print("     model returned nothing twice without hitting the token limit — flaky"
                  " model/serving; re-run before drawing conclusions.")

    print("\n==== SUMMARY " + "=" * 47)
    print(f"  vision capability: {'yes' if vision else 'NO' if vision is False else 'unknown (no /api/show)'}")
    for name, (verdict, detail) in results.items():
        print(f"  {name:<16} {verdict:<8} {detail}")
    if "native-chat" in results and "native-chat-pre" in results:
        print(f"\n  preprocessing A/B:  plain    [{results['native-chat'][0]}] {results['native-chat'][1]}")
        print(f"                      prepped  [{results['native-chat-pre'][0]}] {results['native-chat-pre'][1]}")

    prod = results.get("openai-object", ("SKIPPED", ""))[0]
    natives = [results[n][0] for n in ("native-chat", "native-generate") if n in results]
    all_verdicts = [v for v, _ in results.values()]
    sees = ("PASS", "PASS*", "LOWRES", "LOWRES*")
    print("\n==== VERDICT")
    if vision is False:
        print("  The model tag cannot see images — replace the model (e.g. pull a vision build),")
        print("  then re-run this script. The service code needs no change for this.")
    elif prod == "PASS":
        print("  The production request format (describe_picture) WORKS against this endpoint.")
        print("  No service change needed — point DGS__VLM__BASE_URL/MODEL at this and re-run.")
    elif prod == "PASS*":
        print("  The endpoint DELIVERS the image on the production path, but this thinking model")
        print("  spends the token budget on hidden reasoning: the visible answer gets cut off.")
        print("  Fix in the service, not the encoding: much higher DGS__VLM__MAX_TOKENS, or")
        print("  disable thinking for the VLM (served-side), or use a non-thinking vision model.")
    elif prod in ("LOWRES", "LOWRES*"):
        print("  Transport is FINE: the image reaches the model on the production path. But the")
        print("  model cannot read the diagram's fine print (hostnames/IPs) — descriptions would")
        print("  be vague or hallucinated. Use a stronger vision model (one that PASSes here).")
    elif prod in ("BLIND", "ERROR") and any(v in sees for v in natives):
        print("  Ollama's OpenAI-compat layer drops/mangles the image here, but the NATIVE API sees it.")
        print("  Fix options: upgrade Ollama on the server, or adapt describe.py to POST /api/chat")
        print("  with messages[0].images=[<raw base64>] when the endpoint is native Ollama.")
    elif prod in ("UNCLEAR", "TRUNCATED") and any(v in sees for v in natives):
        print("  Inconclusive on the production path while the native API sees the image — likely")
        print("  weak vision fidelity or answer variance, not transport. Compare the reply texts")
        print("  above; re-run, and prefer a model that PASSes on the production path.")
    elif prod != "PASS" and results.get("openai-string", ("", ""))[0] in sees:
        print("  Only the bare-string image_url form works — adapt the payload in describe.py.")
    elif all_verdicts and all(v == "TRUNCATED" for v in all_verdicts):
        print("  Every variant ran out of tokens before a visible answer — re-run with a much")
        print("  higher --max-tokens (e.g. 8000); no vision conclusion can be drawn yet.")
    else:
        print("  Nothing saw the image. Check: Ollama version (/api/version), whether the tag is a")
        print("  text-only quantization, and server logs while this script runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
