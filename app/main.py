"""
FastAPI 서버 — 한국어 회의 메모 요약 서비스

엔드포인트:
  - GET  /health   : 서버/모델 상태 확인 (인증 불필요)
  - POST /predict  : 텍스트 요약 (API Key 인증 필요)
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.auth import verify_api_key
from app.schemas import (
    SummarizeRequest, SummarizeResponse,
    EvaluateRequest, EvaluateResponse,
    VerifyKeyRequest, VerifyKeyResponse,
)
from app.model_service import (
    load_model, predict, evaluate, verify_gemini_key,
    T5_MODEL_NAME, LLM_MODEL_NAME, CLOUD_MODEL_NAME, EVAL_MODEL_NAME,
)

# ===== 앱 생성 =====
app = FastAPI(
    title="한국어 회의 메모 요약 서비스",
    description="긴 한국어 업무 문단/회의 메모를 핵심만 짧게 요약해주는 API (인증 필요)",
    version="1.0.0",
)

# 로컬 실습용 CORS 허용 (Streamlit 등 다른 출처에서의 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 추론을 별도 스레드에서 돌리기 위한 실행기 (서버가 멈추지 않게)
inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="summary")

# 모델은 전역 변수에 보관 (서버 시작 시 한 번만 로드)
summarizer = None


@app.on_event("startup")
async def startup():
    """서버가 켜질 때 빠름(t5) 모델을 한 번 로드합니다.
    (정확 모드 LLM 은 무거우므로 첫 요청 때 지연 로드됩니다.)"""
    global summarizer
    print(f"모델 로드 중: {T5_MODEL_NAME}")
    summarizer = load_model()
    print("모델 로드 완료")


@app.get("/health", tags=["System"])
async def health_check():
    """서버와 모델 상태를 확인합니다."""
    return {
        "status": "ok" if summarizer is not None else "loading",
        "model_loaded": summarizer is not None,
        "fast_model": T5_MODEL_NAME,        # 빠름 모드 (서버 시작 시 로드)
        "accurate_model": LLM_MODEL_NAME,   # 정확 모드 (첫 요청 시 로드)
        "cloud_model": CLOUD_MODEL_NAME,    # 최고 모드 (Gemini)
        "eval_model": EVAL_MODEL_NAME,      # 평가 모드 (Gemini Pro)
    }


@app.post("/predict", response_model=SummarizeResponse, tags=["Summarize"])
async def summarize(
    request: SummarizeRequest,
    user: str = Depends(verify_api_key),   # 인증 통과 시 사용자 이름이 들어옴
):
    """텍스트를 받아 요약 결과를 반환합니다 (API Key 필요)."""
    if summarizer is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    try:
        loop = asyncio.get_event_loop()
        # 무거운 추론을 별도 스레드에서 실행 (run_in_executor)
        result = await loop.run_in_executor(
            inference_executor,
            predict,
            summarizer,
            request.text,
            request.mode,
            request.max_length,
            request.min_length,
            request.gemini_api_key,
        )
    except Exception as e:
        # 모델 추론 중 오류 → 500
        raise HTTPException(status_code=500, detail=f"요약 처리 실패: {str(e)}")

    return SummarizeResponse(**result)


@app.post("/verify_key", response_model=VerifyKeyResponse, tags=["System"])
async def verify_key(
    request: VerifyKeyRequest,
    user: str = Depends(verify_api_key),
):
    """Gemini API 키가 유효한지 확인합니다."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(inference_executor, verify_gemini_key, request.gemini_api_key)
    return VerifyKeyResponse(**result)


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluate"])
async def evaluate_summaries(
    request: EvaluateRequest,
    user: str = Depends(verify_api_key),   # 인증 필요
):
    """세 방식의 요약을 Gemini 로 평가합니다 (API Key + GEMINI_API_KEY 필요)."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            inference_executor,
            evaluate,
            request.text,
            request.summaries,
            request.gemini_api_key,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"평가 처리 실패: {str(e)}")

    return EvaluateResponse(**result)
