"""
Streamlit 프론트엔드 — 한국어 요약 3가지 방식 비교 + 평가

입력 1개 → 빠름/정확/최고 세 방식으로 각각 요약 → Gemini 가 평가.

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

# 비교할 3가지 방식: (표시이름, 서버 mode 값, 모델명)
MODES = [
    ("⚡ 빠름", "fast", "t5-base"),
    ("🎯 정확", "accurate", "Qwen2.5-3B"),
    ("🏆 최고", "cloud", "Gemini 2.5 Flash"),
]

st.set_page_config(page_title="요약 방식 비교", page_icon="📝", layout="wide")

# ===== 예시 문장 =====
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
    st.session_state["input_text"] = text


if "input_text" not in st.session_state:
    st.session_state["input_text"] = EXAMPLES["📋 회의록"]

# 좁은 비교 카드에 맞게 지표(metric) 숫자/라벨 글자 크기를 줄인다.
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.0rem; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== 헤더 =====
st.title("📝 한국어 요약 — 3가지 방식 비교 + 평가")
st.markdown("입력 하나로 **빠름·정확·최고** 세 방식의 요약을 만들고, **Gemini**가 채점·비교합니다.")

# ===== 사이드바 =====
with st.sidebar:
    st.header("⚙️ 설정")
    api_base = st.text_input("API 주소", value=DEFAULT_API)
    api_key = st.text_input("API Key (서비스 인증)", value=DEFAULT_KEY, type="password")
    gemini_key = st.text_input(
        "Google Gemini API Key (최고/평가 모드용)",
        value="",
        type="password",
        help="입력하면 🏆 최고 모드와 Gemini 평가가 이 키로 동작합니다. "
             "비워두면 서버의 GEMINI_API_KEY를 사용합니다. "
             "(키는 https://aistudio.google.com 에서 무료 발급)",
    )
    # 키 유효성 확인 버튼
    if st.button("🔑 키 확인", use_container_width=True):
        with st.spinner("키 확인 중..."):
            try:
                vr = requests.post(
                    f"{api_base}/verify_key",
                    json={"gemini_api_key": gemini_key},
                    headers={"X-API-Key": api_key},
                    timeout=30,
                )
                if vr.status_code == 200 and vr.json().get("valid"):
                    st.success("✅ 유효한 Gemini 키입니다.")
                elif vr.status_code == 200:
                    st.error(f"❌ 키가 유효하지 않습니다.\n{vr.json().get('detail', '')[:200]}")
                else:
                    st.error(f"확인 실패({vr.status_code})")
            except Exception as e:
                st.error(f"확인 중 오류: {e}")

    try:
        h = requests.get(f"{api_base}/health", timeout=3).json()
        if h.get("model_loaded"):
            st.success("🟢 서버 연결됨 · 모델 준비 완료")
        else:
            st.warning("🟡 서버는 켜졌지만 모델 로딩 중")
    except Exception:
        st.error("🔴 서버에 연결할 수 없습니다.\n터미널에서 `uvicorn app.main:app` 실행 여부를 확인하세요.")

    st.divider()
    with st.expander("🔧 고급 설정 (빠름 모드 길이)"):
        auto_length = st.checkbox("길이 자동 조절", value=True)
        max_length = st.slider("요약 최대 길이", 30, 200, 150, disabled=auto_length)
        min_length = st.slider("요약 최소 길이", 5, 100, 40, disabled=auto_length)

    st.caption("🎯 정확 모드는 CPU에서 1~3분 걸립니다. "
               "🏆 최고 모드와 평가는 Gemini API 키가 있어야 동작합니다 (무료 등급).")

# ===== 입력 =====
st.subheader("1️⃣ 요약할 내용 입력")
cols = st.columns(len(EXAMPLES))
for col, (label, sample) in zip(cols, EXAMPLES.items()):
    col.button(label, use_container_width=True, on_click=load_example, args=(sample,))

text = st.text_area(
    "요약할 원문",
    key="input_text",
    height=200,
    label_visibility="collapsed",
    placeholder="여기에 요약할 글을 붙여넣으세요 (30~5000자)",
)
st.caption(f"📏 {len(text)} / 5000자")

# ===== 실행 =====
st.subheader("2️⃣ 모델 별 비교 실행")
run = st.button("🔍 3가지 방식 비교하고 평가받기", type="primary", use_container_width=True)

if run:
    if not text or len(text.strip()) < 30:
        st.warning("원문은 최소 30자 이상 입력해 주세요.")
    elif len(text) > 5000:
        st.warning("원문은 5000자 이하로 입력해 주세요.")
    else:
        headers = {"X-API-Key": api_key}
        results = {}  # 표시이름 -> {"summary":..., "elapsed":..., "model":...} 또는 {"error":...}

        # ── 세 방식을 차례대로 실행 ──
        for label, mode, model in MODES:
            with st.spinner(f"{label} ({model}) 요약 중..."):
                payload = {"text": text, "mode": mode}
                if mode == "fast" and not auto_length:
                    payload["max_length"] = max_length
                    payload["min_length"] = min_length
                if mode == "cloud" and gemini_key:
                    payload["gemini_api_key"] = gemini_key
                t0 = time.perf_counter()
                try:
                    resp = requests.post(f"{api_base}/predict", json=payload, headers=headers, timeout=300)
                    dt = time.perf_counter() - t0
                    if resp.status_code == 200:
                        d = resp.json()
                        results[label] = {"summary": d["summary"], "elapsed": dt, "model": d["model_name"]}
                    elif resp.status_code == 401:
                        results[label] = {"error": "인증 실패 (API Key 확인)"}
                    elif resp.status_code == 422:
                        results[label] = {"error": "입력값 오류"}
                    else:
                        detail = resp.json().get("detail", "")
                        results[label] = {"error": f"오류({resp.status_code}): {detail}"}
                except requests.exceptions.ConnectionError:
                    results[label] = {"error": "서버에 연결할 수 없습니다."}
                except Exception as e:
                    results[label] = {"error": str(e)}

        # ── Gemini 평가 (카드에 점수를 넣기 위해 먼저 실행) ──
        ok_summaries = {label: r["summary"] for label, r in results.items() if "summary" in r}
        eval_by_name = {}   # 이름 -> 항목별 점수 dict
        best, reason, eval_error = "", "", None
        if len(ok_summaries) >= 2:
            with st.spinner("Gemini가 세 요약을 채점·비교하는 중..."):
                try:
                    eval_payload = {"text": text, "summaries": ok_summaries}
                    if gemini_key:
                        eval_payload["gemini_api_key"] = gemini_key
                    er = requests.post(f"{api_base}/evaluate", json=eval_payload, headers=headers, timeout=180)
                    if er.status_code == 200:
                        data = er.json()
                        eval_by_name = {e.get("name"): e for e in data.get("evaluations", [])}
                        best, reason = data.get("best", ""), data.get("reason", "")
                    elif er.status_code == 401:
                        eval_error = "인증 실패 (API Key 확인)"
                    else:
                        eval_error = f"평가 실패({er.status_code}): {er.json().get('detail', '')}"
                except requests.exceptions.ConnectionError:
                    eval_error = "서버에 연결할 수 없습니다."
                except Exception as e:
                    eval_error = f"평가 오류: {e}"
        else:
            eval_error = ("평가하려면 최소 2개 방식의 요약이 성공해야 합니다. "
                          "(최고 모드 실패 시 사이드바의 Gemini API Key를 확인하세요)")

        # ── 3개 결과를 나란히 표시 (요약 + 지표 + Opus 점수) ──
        st.subheader("3️⃣ 요약 결과 비교")
        result_cols = st.columns(len(MODES))
        for col, (label, mode, model) in zip(result_cols, MODES):
            with col:
                st.markdown(f"#### {label}")
                st.caption(model)
                r = results.get(label, {})
                if "summary" in r:
                    with st.container(border=True):
                        st.write(r["summary"])
                    summ_len = len(r["summary"])
                    # 요약율 = 요약문 길이 / 원문 길이 (원문 대비 몇 %로 줄었는지)
                    ratio = round(summ_len / len(text) * 100) if len(text) else 0
                    c1, c2, c3 = st.columns(3)
                    c1.metric("글자 수", f"{summ_len} 자")
                    c2.metric("요약율", f"{ratio} %")
                    c3.metric("소요", f"{r['elapsed']:.1f} 초")

                    # Gemini 평가 점수 (있을 때만)
                    sc = eval_by_name.get(label)
                    if sc:
                        winner = "🏅 " if best == label else ""
                        st.markdown(f"**🧑‍⚖️ Gemini 평가** {winner}")
                        e1, e2 = st.columns(2)
                        e1.metric("정확성", f"{sc.get('정확성', '-')}/5")
                        e2.metric("핵심포착", f"{sc.get('핵심포착', '-')}/5")
                        e3, e4 = st.columns(2)
                        e3.metric("간결성", f"{sc.get('간결성', '-')}/5")
                        e4.metric("자연스러움", f"{sc.get('자연스러움', '-')}/5")
                        if sc.get("총평"):
                            st.caption(f"💬 {sc['총평']}")
                else:
                    st.error(r.get("error", "실패"))

        # ── 종합 평가 (최고 선정 + 이유) ──
        st.subheader("4️⃣ 🧑‍⚖️ Gemini 종합 평가")
        if eval_error:
            st.warning(eval_error)
        elif best or reason:
            if best:
                st.success(f"🏅 가장 좋은 요약: **{best}**")
            if reason:
                st.markdown(reason)
            st.caption("평가 모델: gemini-2.5-flash")
