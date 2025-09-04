# app/api.py (rút gọn – phần ENV & backend giữ như hiện tại của bạn)
from __future__ import annotations
import os, re, requests
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT
from rag.retriever import retrieve

load_dotenv()
TRANSFORMERS_CKPT = os.getenv("TRANSFORMERS_CKPT", "").strip()
USE_OLLAMA        = os.getenv("USE_OLLAMA", "true").lower() == "true"
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MAX_NEW_TOKENS    = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE       = float(os.getenv("TEMPERATURE", "0.2"))

# ==== init backends (giữ nguyên như bản của bạn) ====
_hf_tok = _hf_model = None
_backend = "ollama"
if TRANSFORMERS_CKPT and (TRANSFORMERS_CKPT.lower().startswith("qwen/") or os.path.isdir(TRANSFORMERS_CKPT)):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    _hf_tok   = AutoTokenizer.from_pretrained(TRANSFORMERS_CKPT, use_fast=True)
    _hf_model = AutoModelForCausalLM.from_pretrained(TRANSFORMERS_CKPT, torch_dtype="auto", device_map="auto")
    if getattr(_hf_tok, "pad_token", None) is None:
        _hf_tok.pad_token = _hf_tok.eos_token
    _backend = "transformers"
elif USE_OLLAMA:
    _backend = "ollama"
else:
    _backend = "none"

app = FastAPI(title="IU Chatbot API (IT/CS/DS)")

class ChatReq(BaseModel):
    query: str
    top_k: int = Field(6, ge=1, le=20)

# ===== helpers =====
def _is_english(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

def _sanitize_output(s: str, lang: str) -> str:
    # tránh gạch ngang do "~"
    s = s.replace("~", "≈")
    # loại ký tự CJK nếu cần (phòng hờ model lỡ sinh CN)
    if lang in ("vi", "en"):
        s = re.sub(r"[\u3040-\u30FF\u3400-\u9FFF\uF900-\uFAFF\uAC00-\uD7AF]+", "", s)
    return s.strip()

def _to_prompt(messages: List[Dict[str, str]]) -> str:
    sys = ""; usr = ""
    for m in messages:
        if m["role"] == "system": sys = m["content"]
        elif m["role"] == "user": usr = m["content"]
    return f"System:\n{sys}\n\nUser:\n{usr}\n\nAssistant:"

def _gen_hf(messages: List[Dict[str, str]]) -> str:
    import torch
    sys = ""; usr = ""
    for m in messages:
        if m["role"] == "system": sys = m["content"]
        elif m["role"] == "user": usr = m["content"]
    prompt = f"<system>\n{sys}\n</system>\n<user>\n{usr}\n</user>\n<assistant>\n"
    inputs = _hf_tok(prompt, return_tensors="pt").to(_hf_model.device)
    with torch.no_grad():
        out = _hf_model.generate(
            **inputs,
            do_sample=TEMPERATURE > 0,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=_hf_tok.eos_token_id,
        )
    text = _hf_tok.decode(out[0], skip_special_tokens=True)
    if "<assistant>" in text:
        text = text.split("<assistant>", 1)[-1]
    return text.strip()

def _gen_ollama(messages: List[Dict[str, str]]) -> str:
    # /api/chat trước → nếu 404 thì /api/generate
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat",
                          json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
                          timeout=15)
        if r.status_code == 200:
            return r.json()["message"]["content"]
        if r.status_code != 404:
            r.raise_for_status()
    except requests.RequestException:
        pass
    prompt = _to_prompt(messages)
    r2 = requests.post(f"{OLLAMA_BASE_URL}/api/generate",
                       json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                       timeout=15)
    r2.raise_for_status()
    return (r2.json().get("response") or "").strip()

@app.get("/")
def root():
    return {"ok": True, "backend": _backend, "use": "POST /chat"}

@app.post("/chat")
def chat(req: ChatReq):
    if _backend == "none":
        return JSONResponse(status_code=500, content={"error": "No backend. Set TRANSFORMERS_CKPT or USE_OLLAMA=true."})

    # 1) RAG
    ctxs = retrieve(req.query, k=req.top_k)
    context = "\n\n".join([c[0] for c in ctxs]) if ctxs else ""
    lang = "en" if _is_english(req.query) else "vi"

    # 2) Nếu không có context → trả fallback NGAY (không gọi LLM để tránh rơi vào tiếng Trung)
    if not context.strip():
        ans = ("Sorry, I don’t have this in the [Context] yet. Please contact A1.610 (cse@hcmiu.edu.vn)."
               if lang == "en"
               else "Xin lỗi, hiện mình chưa có thông tin trong [Context] cho câu hỏi này. "
                    "Bạn có thể liên hệ A1.610 (cse@hcmiu.edu.vn).")
        return {"answer": ans, "sources": [], "backend": _backend}

    # 3) Có context → gọi LLM nhưng khóa ngôn ngữ qua system + sanitize đầu ra
    sys_guard = (
        "Luôn trả lời đúng NGÔN NGỮ người dùng. Với tiếng Việt, tuyệt đối không sinh ký tự tiếng Trung."
        if lang == "vi" else
        "Always answer in English only. Do not use Chinese characters."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + sys_guard},
        {"role": "user", "content": f"[Context]\n{context}\n\n[Question]\n{req.query}"}
    ]
    try:
        answer = _gen_hf(messages) if _backend == "transformers" else _gen_ollama(messages)
    except Exception as e:
        snippet = (ctxs[0][0][:400] + "…") if ctxs else ""
        answer = (f"Model backend error ({e}).\n\nSummary from Context:\n{snippet}"
                  if lang == "en" else
                  f"Model backend lỗi ({e}).\n\nTóm tắt từ Context:\n{snippet}")

    answer = _sanitize_output(answer, lang)
    sources = [meta or {} for _, meta in ctxs]
    return {"answer": answer, "sources": sources, "backend": _backend}
