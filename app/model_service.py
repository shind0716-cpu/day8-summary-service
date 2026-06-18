"""
모델 로드 + 요약 추론 (세 가지 모드)

- 빠름(fast)    : t5-base 요약 모델. 1~3초로 빠르지만 앞부분 위주로 요약.
- 정확(accurate): Qwen2.5-3B 지시형 LLM(로컬). 1~3분 느리지만 핵심을 잘 잡음.
- 최고(cloud)   : Google Gemini(클라우드). 빠르고 품질 좋음. 무료 등급 API 키 필요.

함수:
- load_model()  : 빠름 모델(t5)을 서버 시작 시 1번 로드
- predict(...)  : mode 에 따라 t5 / Qwen / Gemini 로 요약하고 결과 dict 반환
"""
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ── 모델 이름 ─────────────────────────────────────────────
T5_MODEL_NAME = "eenzeenee/t5-base-korean-summarization"     # 빠름
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"                  # 정확
CLOUD_MODEL_NAME = "gemini-2.5-flash"                        # 최고 (Gemini, 무료 등급)
EVAL_MODEL_NAME = "gemini-2.5-flash"                         # 평가 (Pro는 무료 한도 0이라 Flash 사용)

# 이 T5 모델은 입력 앞에 "summarize: " 접두어를 붙여 학습되었습니다.
PREFIX = "summarize: "

# 요약 비서에게 줄 공통 지시문 (정확/최고 모드 공용)
SUMMARY_SYSTEM = ("너는 한국어 글을 요약하는 비서다. 핵심 내용과 결정사항을 중심으로 간결하게 요약하라. "
                  "입력이 이미 짧으면 한 문장으로 핵심만 요약하라.")

# 무거운 자원은 처음 요청이 들어올 때 한 번만 로드(지연 로드)합니다.
_llm = None       # 정확 모드: (tokenizer, model) 캐시
_gemini = None    # 최고/평가 모드: Gemini 클라이언트 캐시(환경변수 키로 만든 경우)


def load_model():
    """빠름(t5) 요약 파이프라인을 로드해서 반환합니다 (서버 시작 시 1회)."""
    return pipeline("summarization", model=T5_MODEL_NAME)


def _get_llm():
    """정확 모드 LLM 을 (처음 한 번만) 로드해서 반환합니다."""
    global _llm
    if _llm is None:
        torch.set_num_threads(10)   # CPU 코어를 최대한 사용
        tok = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.bfloat16)
        model.eval()
        _llm = (tok, model)
    return _llm


# ── 빠름 모드: 입력 길이에 맞춘 자동 길이 ──────────────────
def auto_lengths(model, text: str):
    """입력 토큰 수에 맞춰 t5 요약 길이(max/min)를 자동 계산."""
    n = len(model.tokenizer.encode(text))
    max_length = min(200, n + 10)                  # 넉넉히(문장 잘림 방지)
    min_length = max(10, min(80, round(n * 0.3)))  # 입력의 약 30%
    return max_length, min_length


def _summarize_t5(model, text, max_length, min_length):
    if max_length is None or min_length is None:
        auto_max, auto_min = auto_lengths(model, text)
        max_length = auto_max if max_length is None else max_length
        min_length = auto_min if min_length is None else min_length
    min_length = min(min_length, max_length - 5)
    result = model(
        PREFIX + text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
        num_beams=6,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
    )
    return result[0]["summary_text"]


# ── 정확 모드: 지시형 LLM (로컬 Qwen) ─────────────────────
def _summarize_llm(text):
    tok, model = _get_llm()
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"다음 글을 요약해줘:\n\n{text}"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1)
    # 새로 생성된 부분만 잘라서 디코딩
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ── 최고/평가 모드: Google Gemini ─────────────────────────
def _get_gemini(api_key: str = None):
    """Gemini 클라이언트를 만듭니다.

    - api_key 가 주어지면(예: 화면에서 입력) 그 키로 클라이언트를 만듭니다.
    - 없으면 서버의 GEMINI_API_KEY / GOOGLE_API_KEY 환경변수를 사용합니다.
    - 둘 다 없으면 친절한 안내 메시지와 함께 오류를 냅니다.
    """
    from google import genai
    import os

    if api_key:                                  # 화면에서 받은 키 우선 사용
        return genai.Client(api_key=api_key)

    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not env_key:
        raise RuntimeError(
            "최고/평가 모드를 쓰려면 Gemini API 키가 필요합니다. "
            "화면 사이드바에 Google Gemini API Key를 입력하거나, 서버에 GEMINI_API_KEY를 설정하세요. "
            "(키는 https://aistudio.google.com 에서 무료로 발급)"
        )

    global _gemini
    if _gemini is None:
        _gemini = genai.Client(api_key=env_key)
    return _gemini


def _summarize_cloud(text, api_key=None):
    from google.genai import types
    client = _get_gemini(api_key)
    resp = client.models.generate_content(
        model=CLOUD_MODEL_NAME,
        contents=f"다음 글을 요약해줘:\n\n{text}",
        config=types.GenerateContentConfig(system_instruction=SUMMARY_SYSTEM),
    )
    return (resp.text or "").strip()


def verify_gemini_key(api_key: str = None) -> dict:
    """Gemini 키가 유효한지 가볍게 확인합니다(모델 목록 조회 — 생성 토큰 소모 없음)."""
    try:
        client = _get_gemini(api_key)
        models = client.models.list()
        next(iter(models), None)   # 실제 호출을 강제해서 키 유효성 확인
        return {"valid": True, "detail": "키가 유효합니다."}
    except Exception as e:
        return {"valid": False, "detail": str(e)}


# ── 평가: Gemini Pro 가 세 요약을 채점 ────────────────────
def evaluate(text: str, summaries: dict, api_key: str = None) -> dict:
    """원문과 세 방식의 요약을 Gemini 로 평가하고, 항목별 점수(JSON)로 반환."""
    import json
    from google.genai import types

    client = _get_gemini(api_key)
    names = list(summaries.keys())
    labeled = "\n\n".join(f"[{name}]\n{summary}" for name, summary in summaries.items())

    prompt = (
        "다음은 한 원문에 대한 여러 요약 결과다. 각 요약을 아래 4개 항목으로 1~5점(정수) 채점해줘.\n"
        "- 정확성: 원문 내용을 왜곡 없이 담았는가\n"
        "- 핵심포착: 가장 중요한 내용(특히 결정사항)을 담았는가\n"
        "- 간결성: 불필요한 군더더기 없이 짧은가\n"
        "- 자연스러움: 한국어 문장이 매끄러운가\n\n"
        "반드시 아래 형식의 JSON만 출력해. 설명·코드블록 없이 JSON 객체 하나만.\n"
        '{\n'
        '  "evaluations": [\n'
        '    {"name": "요약이름", "정확성": 5, "핵심포착": 4, "간결성": 3, "자연스러움": 5, "총평": "한 줄 평"}\n'
        '  ],\n'
        '  "best": "가장 좋은 요약 이름",\n'
        '  "reason": "그 요약을 고른 이유"\n'
        '}\n'
        f'"name" 은 반드시 다음 값들을 그대로 사용해: {names}\n\n'
        f"=== 원문 ===\n{text}\n\n"
        f"=== 요약들 ===\n{labeled}"
    )
    resp = client.models.generate_content(
        model=EVAL_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    raw = (resp.text or "").strip()

    # 혹시 ```json ... ``` 코드블록으로 감싸져 오면 벗겨낸다.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):raw.rfind("}") + 1]

    try:
        data = json.loads(raw)
        evaluations = data.get("evaluations", [])
        best = data.get("best", "")
        reason = data.get("reason", "")
    except Exception:
        # JSON 파싱 실패 시: 점수는 비우고 원문 텍스트를 이유에 담아 둔다.
        evaluations, best, reason = [], "", raw

    return {
        "success": True,
        "model_name": EVAL_MODEL_NAME,
        "evaluations": evaluations,
        "best": best,
        "reason": reason,
    }


def predict(model, text: str, mode: str = "fast",
            max_length: int = None, min_length: int = None,
            api_key: str = None) -> dict:
    """
    mode 에 따라 요약을 수행하고 결과를 dict 로 반환합니다.

    - mode="fast"     : t5 모델 (빠름). max_length/min_length 미지정 시 입력 길이에 맞춰 자동.
    - mode="accurate" : Qwen LLM (정확). 길이는 모델이 알아서 조절.
    - mode="cloud"    : Gemini (최고). api_key 가 있으면 그 키로 호출.
    """
    if mode == "cloud":
        summary = _summarize_cloud(text, api_key)
        model_name = CLOUD_MODEL_NAME
    elif mode == "accurate":
        summary = _summarize_llm(text)
        model_name = LLM_MODEL_NAME
    else:
        summary = _summarize_t5(model, text, max_length, min_length)
        model_name = T5_MODEL_NAME

    return {
        "success": True,
        "summary": summary,
        "original_length": len(text),
        "summary_length": len(summary),
        "model_name": model_name,
    }
