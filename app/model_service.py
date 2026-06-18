"""
모델 로드 + 요약 추론 (두 가지 모드)

- 빠름(fast)   : t5-base 요약 모델. 1~3초로 빠르지만 앞부분 위주로 요약.
- 정확(accurate): Qwen2.5-3B 지시형 LLM. 1~3분 느리지만 결정사항 등 핵심을 잘 잡음.

함수:
- load_model()  : 빠름 모델(t5)을 서버 시작 시 1번 로드
- predict(...)  : mode 에 따라 t5 또는 LLM 으로 요약하고 결과 dict 반환
"""
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ── 모델 이름 ─────────────────────────────────────────────
T5_MODEL_NAME = "eenzeenee/t5-base-korean-summarization"     # 빠름
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"                  # 정확

# 이 T5 모델은 입력 앞에 "summarize: " 접두어를 붙여 학습되었습니다.
PREFIX = "summarize: "

# 정확 모드 LLM 은 무거우므로, 처음 요청이 들어올 때 한 번만 로드(지연 로드)합니다.
_llm = None   # (tokenizer, model) 를 담아둘 캐시


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


# ── 정확 모드: 지시형 LLM ─────────────────────────────────
def _summarize_llm(text):
    tok, model = _get_llm()
    messages = [
        {"role": "system",
         "content": "너는 한국어 글을 요약하는 비서다. 핵심 내용과 결정사항을 중심으로 간결하게 요약하라. "
                    "입력이 이미 짧으면 한 문장으로 핵심만 요약하라."},
        {"role": "user", "content": f"다음 글을 요약해줘:\n\n{text}"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1)
    # 새로 생성된 부분만 잘라서 디코딩
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def predict(model, text: str, mode: str = "fast",
            max_length: int = None, min_length: int = None) -> dict:
    """
    mode 에 따라 요약을 수행하고 결과를 dict 로 반환합니다.

    - mode="fast"     : t5 모델 (빠름). max_length/min_length 미지정 시 입력 길이에 맞춰 자동.
    - mode="accurate" : Qwen LLM (정확). 길이는 모델이 알아서 조절.
    """
    if mode == "accurate":
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
