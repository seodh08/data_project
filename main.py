import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="운동 데이터 분석", layout="wide")

st.title("🏋️ 운동 데이터 분석 웹페이지")

uploaded_file = st.file_uploader(
    "운동 데이터 엑셀 또는 CSV 파일 업로드",
    type=["xlsx", "csv"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith("xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
else:
    st.warning("파일을 업로드해주세요.")
    st.stop()

# 데이터 미리보기
st.subheader("데이터 미리보기")
st.dataframe(df.head())

# 숫자형 컬럼만 선택
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# 체지방 컬럼 선택
bodyfat_col = st.selectbox(
    "체지방률 컬럼 선택",
    numeric_cols
)

# 상관관계 계산
corr = df[numeric_cols].corr()

st.subheader("📊 체지방률과의 상관관계")
corr_with_bodyfat = corr[bodyfat_col].drop(bodyfat_col).sort_values(key=abs, ascending=False)
st.dataframe(corr_with_bodyfat)

top_feature = corr_with_bodyfat.index[0]
st.success(
    f"체지방률과 가장 상관관계가 높은 속성은 **{top_feature}** "
    f"(상관계수: {corr_with_bodyfat.iloc[0]:.3f})"
)

# 산점도
st.subheader("📈 산점도")
x_col = st.selectbox("비교할 속성 선택", corr_with_bodyfat.index)

fig, ax = plt.subplots()
ax.scatter(df[x_col], df[bodyfat_col])
ax.set_xlabel(x_col)
ax.set_ylabel(bodyfat_col)
ax.set_title(f"{x_col} vs {bodyfat_col}")
st.pyplot(fig)

# 히트맵
st.subheader("🔥 상관관계 히트맵")
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)
