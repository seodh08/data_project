import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(
    page_title="날씨와 대중교통 이용량 분석",
    layout="wide"
)
st.title("🚇 기상상태와 대중교통 이용량 간의 상관관계 분석")

# ---------------------------
# 데이터 로드
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "한국교통안전공단_대중교통 기상상태별 이용인원_20221231.csv",
        encoding="cp949"
    )
    return df

df = load_data()

st.subheader("📄 원본 데이터")
st.dataframe(df)

# ---------------------------
# 분석용 데이터
# ---------------------------
weather_cols = ["맑은 날", "강우", "강설"]
analysis_df = df[weather_cols]

# ---------------------------
# 🔹 정규화 (Min-Max Normalization)
# ---------------------------
analysis_df_norm = (
    analysis_df - analysis_df.min()
) / (analysis_df.max() - analysis_df.min())

st.subheader("📐 정규화된 데이터 (0 ~ 1 범위)")
st.dataframe(analysis_df_norm)

# ---------------------------
# 상관계수 계산 (정규화 데이터 기준)
# ---------------------------
corr = analysis_df_norm.corr()

st.subheader("📊 기상상태 간 상관계수 히트맵 (정규화 기준)")

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title="정규화된 기상상태별 대중교통 이용량 상관계수"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 상관관계 상세 분석
# ---------------------------
st.subheader("🔍 상관관계 상세 분석")

corr_pairs = corr.unstack()
corr_pairs = corr_pairs[
    corr_pairs.index.get_level_values(0)
    != corr_pairs.index.get_level_values(1)
]

max_positive = corr_pairs.idxmax()
max_negative = corr_pairs.idxmin()

col1, col2 = st.columns(2)

with col1:
    if st.button("📈 양의 상관관계가 가장 높은 기상 상태"):
        st.success(
            f"""
            **{max_positive[0]} ↔ {max_positive[1]}**

            상관계수: **{corr_pairs[max_positive]:.2f}**
            """
        )

with col2:
    if st.button("📉 음의 상관관계가 가장 높은 기상 상태"):
        st.warning(
            f"""
            **{max_negative[0]} ↔ {max_negative[1]}**

            상관계수: **{corr_pairs[max_negative]:.2f}**
            """
        )
