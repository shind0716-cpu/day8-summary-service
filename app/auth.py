"""
API Key 인증 모듈 (Day 6 방식 재사용)

요청 헤더의 X-API-Key 값을 확인하여,
올바른 키가 아니면 401 Unauthorized 를 반환합니다.
"""
from fastapi import HTTPException, Header

# 실습용 API Key 목록 { 키: 사용자이름 }
# 실제 서비스라면 환경변수나 DB에 보관해야 하지만, 과제에서는 코드에 적어 둡니다.
VALID_API_KEYS = {
    "my-secret-key": "사용자A",
}


async def verify_api_key(x_api_key: str = Header(None)) -> str:
    """
    FastAPI 의 Depends() 에서 사용하는 인증 함수.

    - 헤더에 X-API-Key 가 없으면          → 401
    - X-API-Key 가 등록된 키가 아니면      → 401
    - 올바른 키면 해당 사용자 이름을 반환   → 엔드포인트에서 사용 가능
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key가 필요합니다. X-API-Key 헤더를 포함해 주세요.",
        )
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="유효하지 않은 API Key입니다.",
        )
    return VALID_API_KEYS[x_api_key]
