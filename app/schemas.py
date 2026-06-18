"""
요약 요청 / 응답 스키마 (Pydantic)

- SummarizeRequest  : 클라이언트가 보내는 입력 (검증 규칙 포함)
- SummarizeResponse : 서버가 돌려주는 결과 형태
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    """요약 요청 입력."""

    text: str = Field(
        ...,                      # ... 는 "필수 항목" 이라는 뜻
        min_length=30,            # 너무 짧으면 요약 의미가 없으므로 30자 이상
        max_length=5000,          # 회의 메모도 담을 수 있도록 5000자까지 허용
        description="요약할 한국어 원문 (30~5000자)",
    )
    # 요약 방식: fast(t5, 빠름) / accurate(LLM, 정확하지만 느림)
    mode: Literal["fast", "accurate"] = Field(
        default="fast",
        description="fast=t5(빠름, 1~3초) / accurate=LLM(정확, CPU에서 1~3분)",
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


class SummarizeResponse(BaseModel):
    """요약 결과 응답."""

    success: bool
    summary: str
    original_length: int          # 원문 글자 수
    summary_length: int           # 요약문 글자 수
    model_name: str
