import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="날씨와 대중교통 이용량 분석", layout="wide")
st.title("기상상태와 대중교통 이용량 간의 상관관계 분석")

@st.cache_data
def load_data():
    df = pd.read_csv(
        "한국교통안전공단_대중교통 기상상태별 이용인원_20221231.csv",
        encoding="cp949"
    )
    return df

df = load_data()

st.subheader("원본 데이터")
st.dataframe(df)

weather_cols = ["맑은 날", "강우", "강설"]
analysis_df = df[weather_cols]

corr = analysis_df.corr()

st.subheader("기상상태 간 상관계수 히트맵")

fig, ax = plt.subplots()
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)
st.pyplot(fig)

st.subheader("상관관계 상세 분석")

corr_pairs = corr.unstack()
corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]

max_positive = corr_pairs.idxmax()
max_negative = corr_pairs.idxmin()

col1, col2 = st.columns(2)

with col1:
    if st.button("양의 상관관계가 가장 높은 기상 상태"):
        st.success(
            f"가장 높은 **양의 상관관계**는\n\n"
            f" **{max_positive[0]} ↔ {max_positive[1]}**\n\n"
            f"상관계수: **{corr_pairs[max_positive]:.2f}**"
        )

with col2:
    if st.button("음의 상관관계가 가장 높은 기상 상태"):
        st.warning(
            f"가장 높은 **음의 상관관계**는\n\n"
            f" **{max_negative[0]} ↔ {max_negative[1]}**\n\n"
            f"상관계수: **{corr_pairs[max_negative]:.2f}**"
        )

st.markdown("""
### 해석 가이드
- **양의 상관관계**: 한 기상 상태의 이용량이 증가하면 다른 상태의 이용량도 함께 증가
- **음의 상관관계**: 한 기상 상태의 이용량이 증가할수록 다른 상태의 이용량은 감소
- 상관계수는 **-1 ~ 1** 사이의 값을 가짐
""")
