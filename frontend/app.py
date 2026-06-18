"""
Streamlit 프론트엔드 — 한국어 회의 메모 요약 서비스 (사용자 친화 버전)

실행 방법:
    streamlit run frontend/app.py
(FastAPI 서버가 먼저 실행 중이어야 합니다)
"""
import time

import requests
import streamlit as st

# ===== 기본 설정 =====
DEFAULT_API = "http://127.0.0.1:8000"
DEFAULT_KEY = "my-secret-key"

st.set_page_config(page_title="회의 메모 요약 서비스", page_icon="📝", layout="centered")

# ===== 예시 문장 (버튼으로 불러오기) =====
EXAMPLES = {
    "📋 회의록": (
        "오늘 주간 회의에서는 신제품 출시 일정과 마케팅 예산 배분에 대해 논의하였다. "
        "개발팀은 핵심 기능 구현이 예정보다 일주일 지연되고 있다고 보고하였고, "
        "디자인팀은 최종 시안이 다음 주 화요일에 완료된다고 공유하였다. "
        "마케팅팀은 온라인 광고 비중을 늘리는 대신 오프라인 행사 예산을 축소하는 방안을 제안하였다. "
        "최종적으로 출시일은 2주 늦추되, 사전 예약 이벤트를 먼저 진행하기로 결정하였다."
    ),
    "📢 공지문": (
        "안녕하세요. 총무팀입니다. 다음 주 월요일부터 사무실 출입 시스템이 새 카드로 교체됩니다. "
        "기존 출입카드는 금요일까지만 사용 가능하며, 신규 카드는 각 팀 대표가 목요일 오후에 수령해 배부할 예정입니다. "
        "분실 시 재발급에는 영업일 기준 3일이 소요되니 주의해 주시기 바랍니다."
    ),
    "✉️ 업무 메일": (
        "팀장님께. 어제 요청하신 3분기 매출 보고서 초안을 첨부드립니다. "
        "지난 분기 대비 전체 매출은 12% 증가했으며, 특히 온라인 채널이 크게 성장했습니다. "
        "다만 오프라인 매장 매출은 소폭 감소하여, 다음 회의 때 대응 방안을 논의했으면 합니다. "
        "검토 후 수정 의견 주시면 금요일까지 최종본을 완성하겠습니다."
    ),
}


def load_example(text):
    """예시 버튼을 누르면 입력칸에 채워 넣는다."""
    st.session_state["input_text"] = text


# 입력칸 초기값
if "input_text" not in st.session_state:
    st.session_state["input_text"] = EXAMPLES["📋 회의록"]

# ===== 헤더 =====
st.title("📝 한국어 회의 메모 요약 서비스")
st.markdown("긴 **회의 메모·공지문·업무 메일**을 붙여넣으면 AI가 핵심만 짧게 요약해 드립니다.")

# ===== 사이드바: 상태 + 고급 설정 =====
with st.sidebar:
    st.header("⚙️ 설정")
    api_base = st.text_input("API 주소", value=DEFAULT_API)
    api_key = st.text_input("API Key", value=DEFAULT_KEY, type="password")

    # 서버 연결 상태 표시
    try:
        h = requests.get(f"{api_base}/health", timeout=3).json()
        if h.get("model_loaded"):
            st.success("🟢 서버 연결됨 · 모델 준비 완료")
        else:
            st.warning("🟡 서버는 켜졌지만 모델 로딩 중")
    except Exception:
        st.error("🔴 서버에 연결할 수 없습니다.\n터미널에서 `uvicorn app.main:app` 실행 여부를 확인하세요.")

    st.divider()
    with st.expander("🔧 고급 설정 (요약 길이 직접 지정)"):
        auto_length = st.checkbox("길이 자동 조절 (입력 길이에 맞춤)", value=True)
        max_length = st.slider("요약 최대 길이", 30, 200, 150, disabled=auto_length)
        min_length = st.slider("요약 최소 길이", 5, 100, 40, disabled=auto_length)
        st.caption("자동을 끄면 직접 길이를 정할 수 있습니다. (빠름 모드에만 적용)")

# ===== 요약 방식 선택 =====
st.subheader("1️⃣ 요약 방식 선택")
mode_label = st.radio(
    "요약 방식",
    ["⚡ 빠름 (수 초)", "🎯 정확 (1~3분)"],
    captions=["t5 모델 · 일반적인 글에 적합", "AI 모델 · 결정사항/핵심을 더 잘 잡음"],
    horizontal=True,
    label_visibility="collapsed",
)
mode = "accurate" if mode_label.startswith("🎯") else "fast"

# ===== 입력 =====
st.subheader("2️⃣ 요약할 내용 입력")

# 예시 불러오기 버튼
cols = st.columns(len(EXAMPLES))
for col, (label, sample) in zip(cols, EXAMPLES.items()):
    col.button(label, use_container_width=True, on_click=load_example, args=(sample,))

text = st.text_area(
    "요약할 원문",
    key="input_text",
    height=240,
    label_visibility="collapsed",
    placeholder="여기에 요약할 글을 붙여넣으세요 (30~5000자)",
)

# 글자수 카운터 (색으로 안내)
n = len(text)
if n < 30:
    st.caption(f"✏️ {n} / 5000자 — 최소 30자 이상 입력해 주세요.")
elif n > 5000:
    st.caption(f"⚠️ {n} / 5000자 — 너무 깁니다. 5000자 이하로 줄여 주세요.")
elif mode == "fast" and n > 1000:
    st.caption(f"📏 {n} / 5000자 — 빠름 모드는 약 1000자까지만 처리합니다. 긴 글은 '정확' 모드를 권장합니다.")
else:
    st.caption(f"📏 {n} / 5000자")

# ===== 실행 =====
st.subheader("3️⃣ 요약 실행")
if st.button("✨ 요약하기", type="primary", use_container_width=True):
    if not text or len(text.strip()) < 30:
        st.warning("원문은 최소 30자 이상 입력해 주세요.")
    elif len(text) > 5000:
        st.warning("원문은 5000자 이하로 입력해 주세요.")
    else:
        spin = "🎯 정확 모드로 요약 중입니다... (1~3분 걸릴 수 있어요)" if mode == "accurate" else "⚡ 요약 중입니다..."
        with st.spinner(spin):
            try:
                payload = {"text": text, "mode": mode}
                if mode == "fast" and not auto_length:
                    payload["max_length"] = max_length
                    payload["min_length"] = min_length

                t0 = time.perf_counter()
                resp = requests.post(
                    f"{api_base}/predict",
                    json=payload,
                    headers={"X-API-Key": api_key},
                    timeout=300,
                )
                elapsed = time.perf_counter() - t0   # 요약 소요 시간(초)

                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ 요약 완료! (소요 시간 {elapsed:.1f}초)")

                    # 결과 카드
                    with st.container(border=True):
                        st.markdown("#### 📄 요약 결과")
                        st.write(data["summary"])

                    # 통계
                    orig, summ = data["original_length"], data["summary_length"]
                    ratio = round(summ / orig * 100) if orig else 0
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("원문 길이", f"{orig} 자")
                    c2.metric("요약문 길이", f"{summ} 자")
                    c3.metric("압축률", f"{ratio} %")
                    c4.metric("소요 시간", f"{elapsed:.1f} 초")

                    st.download_button(
                        "💾 요약문 저장 (.txt)",
                        data=data["summary"],
                        file_name="summary.txt",
                        use_container_width=True,
                    )
                    st.caption(f"사용 모델: {data['model_name']}")

                elif resp.status_code == 401:
                    st.error("🔑 인증 실패: API Key가 없거나 올바르지 않습니다. (사이드바에서 확인)")
                elif resp.status_code == 422:
                    st.error("📋 입력값 오류: 글자 수나 설정값이 허용 범위를 벗어났습니다.")
                    with st.expander("자세한 오류 내용"):
                        st.json(resp.json())
                else:
                    st.error(f"서버 오류 (상태 코드 {resp.status_code})")
                    st.json(resp.json())

            except requests.exceptions.ConnectionError:
                st.error("🔴 서버에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인하세요.\n\n"
                         "터미널: `uvicorn app.main:app --reload`")
            except requests.exceptions.Timeout:
                st.error("⏱️ 응답 시간이 초과되었습니다. 정확 모드는 시간이 오래 걸리니 잠시 후 다시 시도해 주세요.")
            except Exception as e:
                st.error(f"알 수 없는 오류: {e}")
