"""
요약 요청 / 응답 스키마 (Pydantic)

- SummarizeRequest  : 클라이언트가 보내는 입력 (검증 규칙 포함)
- SummarizeResponse : 서버가 돌려주는 결과 형태
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    """요약 요청 입력."""

    text: str = Field(
        ...,                      # ... 는 "필수 항목" 이라는 뜻
        min_length=30,            # 너무 짧으면 요약 의미가 없으므로 30자 이상
        max_length=5000,          # 회의 메모도 담을 수 있도록 5000자까지 허용
        description="요약할 한국어 원문 (30~5000자)",
    )
    # 요약 방식: fast(t5) / accurate(Qwen, 로컬) / cloud(Gemini API)
    mode: Literal["fast", "accurate", "cloud"] = Field(
        default="fast",
        description="fast=t5(빠름, 1~3초) / accurate=Qwen(정확, CPU 1~3분) / cloud=Gemini(최고, API 키 필요)",
    )
    # max_length / min_length 를 보내지 않으면(null) 입력 길이에 맞춰 자동으로 정해진다.
    max_length: Optional[int] = Field(
        default=None,
        ge=30, le=200,            # 직접 지정 시 30~200
        description="요약 결과의 최대 토큰 길이 (미지정 시 입력 길이에 맞춰 자동)",
    )
    min_length: Optional[int] = Field(
        default=None,
        ge=5, le=100,             # 직접 지정 시 5~100
        description="요약 결과의 최소 토큰 길이 (미지정 시 입력 길이에 맞춰 자동)",
    )
    # 최고(cloud) 모드용 Gemini 키. 보내면 서버가 이 키로 Gemini를 호출한다.
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API 키 (cloud 모드에서 사용, 미지정 시 서버 환경변수 사용)",
    )


class SummarizeResponse(BaseModel):
    """요약 결과 응답."""

    success: bool
    summary: str
    original_length: int          # 원문 글자 수
    summary_length: int           # 요약문 글자 수
    model_name: str


class EvaluateRequest(BaseModel):
    """세 방식의 요약을 평가해 달라는 요청."""

    text: str = Field(..., min_length=30, max_length=5000, description="요약의 기준이 되는 원문")
    # { "빠름": "요약문", "정확": "요약문", "최고": "요약문" } 형태
    summaries: Dict[str, str] = Field(..., description="방식 이름 → 요약문 매핑")
    # 평가용 Gemini 키. 보내면 서버가 이 키로 호출한다.
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API 키 (평가에 사용, 미지정 시 서버 환경변수 사용)",
    )


class VerifyKeyRequest(BaseModel):
    """Gemini 키 유효성 확인 요청."""

    gemini_api_key: Optional[str] = Field(default=None, description="확인할 Gemini API 키")


class VerifyKeyResponse(BaseModel):
    """키 확인 결과."""

    valid: bool
    detail: str = ""


class EvaluateResponse(BaseModel):
    """평가 결과 응답 (항목별 점수)."""

    success: bool
    model_name: str                       # 평가에 사용한 모델 (Gemini Pro)
    evaluations: List[Dict[str, Any]]     # [{name, 정확성, 핵심포착, 간결성, 자연스러움, 총평}, ...]
    best: str = ""                        # 가장 좋은 요약 이름
    reason: str = ""                      # 선정 이유
