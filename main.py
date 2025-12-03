# Creating Streamlit app files from the uploaded Excel and showing a quick analysis result.
# This code will:
# 1) Load the uploaded Excel at /mnt/data/fitness data.xlsx
# 2) Detect a body-fat-like column automatically (by common Korean/English names or numeric)
# 3) Compute correlations and find the attribute most correlated with body fat percentage
# 4) Save a Streamlit app (app.py) and requirements.txt to /mnt/data/streamlit_fitness_app/
# 5) Display a small preview table and the detected results for you to download the created files.

import os
import io
import pandas as pd
import numpy as np
from caas_jupyter_tools import display_dataframe_to_user

input_path = "/mnt/data/fitness data.xlsx"
out_dir = "/mnt/data/streamlit_fitness_app"
os.makedirs(out_dir, exist_ok=True)

# 1) Load Excel (first sheet)
try:
    df = pd.read_excel(input_path)
except Exception as e:
    raise RuntimeError(f"Couldn't read the uploaded Excel at {input_path}: {e}")

# show preview to user
display_dataframe_to_user("Fitness data (first 200 rows preview)", df.head(200))

# 2) Detect body fat column by checking common keywords (Korean + English)
bodyfat_candidates = []
keywords = ["체지방", "체지방율", "체지방률", "체지방율(%)", "체지방률(%)",
            "body_fat", "bodyfat", "body fat", "body_fat_percent", "bf%", "bf_percent", "body_fat(%)", "bodyfat%"]
for col in df.columns:
    low = str(col).lower()
    for kw in keywords:
        if kw in low:
            bodyfat_candidates.append(col)

# If none found, choose the numeric column with name suggesting fat or fallback to the first numeric column
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
bodyfat_col = None
if bodyfat_candidates:
    bodyfat_col = bodyfat_candidates[0]
elif numeric_cols:
    # try to find columns containing 'fat' in name
    for col in numeric_cols:
        if "fat" in str(col).lower() or "체지방" in str(col):
            bodyfat_col = col
            break
    if bodyfat_col is None:
        # fallback to the numeric column with median-like values between 1 and 60 maybe
        for col in numeric_cols:
            ser = df[col].dropna()
            if len(ser) == 0: 
                continue
            med = ser.median()
            if 1 <= med <= 60:
                bodyfat_col = col
                break
    if bodyfat_col is None:
        bodyfat_col = numeric_cols[0]

# 3) Compute correlations
corr_df = df.select_dtypes(include=[np.number]).corr()
if bodyfat_col not in corr_df.columns:
    top_corr_col = None
    top_corr_val = None
else:
    correlations = corr_df[bodyfat_col].drop(labels=[bodyfat_col], errors='ignore').abs().sort_values(ascending=False)
    if len(correlations) == 0:
        top_corr_col = None
        top_corr_val = None
    else:
        top_corr_col = correlations.index[0]
        top_corr_val = corr_df.loc[top_corr_col, bodyfat_col]

# 4) Create Streamlit app (app.py) and requirements.txt
app_py = f'''# Streamlit app for fitness data analysis
# Save this file as app.py and run with: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(layout="wide", page_title="Fitness Data Analyzer")

st.title("Fitness Data Analyzer")
st.write("Upload an Excel/CSV or use the included data (if available). This app finds which attribute is most correlated with body fat percentage, and shows scatter and heatmap plots.")

uploaded = st.file_uploader("Upload an Excel or CSV file", type=["xlsx","xls","csv"])
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {{e}}")
        st.stop()
else:
    # try to load bundled file next to app if present
    try:
        df = pd.read_excel("fitness data.xlsx")
    except Exception as e:
        st.info("업로드된 파일이 없고 기본 데이터도 찾을 수 없습니다. 파일을 업로드해 주세요.")
        st.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("데이터에 숫자형 컬럼이 없습니다. 숫자형 데이터가 포함된 파일을 업로드해 주세요.")
    st.stop()

# Auto-detect bodyfat column by common names; allow override
candidates = [c for c in df.columns if any(k in str(c).lower() for k in ['체지방','체지방율','체지방률','body_fat','bodyfat','bf'])]
default_bf = candidates[0] if candidates else (numeric_cols[0] if numeric_cols else None)
bodyfat = st.selectbox("체지방(Body fat)로 사용할 컬럼을 선택하세요", options=numeric_cols, index=numeric_cols.index(default_bf) if default_bf in numeric_cols else 0)

# show correlations
corr = df.select_dtypes(include=[np.number]).corr()
st.subheader("상관관계 (Correlation with selected body fat column)")
corr_with_bf = corr[bodyfat].drop(labels=[bodyfat], errors='ignore').sort_values(key=abs, ascending=False)
st.table(corr_with_bf.to_frame(name='corr'))

if len(corr_with_bf) > 0:
    top_attr = corr_with_bf.index[0]
    st.markdown(f"**체지방과 가장 상관관계가 높은 속성:** {top_attr} (상관계수 = {corr_with_bf.iloc[0]:.3f})")

# Scatter plot selector
st.subheader("산점도 (Scatter plot)")
x_col = st.selectbox("X 축으로 사용할 속성 선택", options=[c for c in numeric_cols if c != bodyfat], index=0 if any(c!=bodyfat for c in numeric_cols) else 0)
sample_size = st.slider("샘플 수 (플롯에 표시할 최대 행 수)", 50, 10000, 1000, step=50)

plot_df = df[[bodyfat, x_col]].dropna().head(sample_size)
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(plot_df[x_col], plot_df[bodyfat])
# regression line
try:
    m, b = np.polyfit(plot_df[x_col], plot_df[bodyfat], 1)
    xs = np.array([plot_df[x_col].min(), plot_df[x_col].max()])
    ax.plot(xs, m*xs + b)
except Exception:
    pass
ax.set_xlabel(str(x_col))
ax.set_ylabel(str(bodyfat))
ax.set_title(f"Scatter: {{x_col}} vs {{bodyfat}}")
st.pyplot(fig)

# Correlation heatmap
st.subheader("히트맵 (Correlation matrix)")
cols_for_heatmap = st.multiselect("히트맵에 포함할 숫자형 컬럼 선택 (최대 30개)", options=numeric_cols, default=numeric_cols[:20])
if cols_for_heatmap:
    cm = df[cols_for_heatmap].corr()
    fig2, ax2 = plt.subplots(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='.2f', ax=ax2, cmap='vlag', center=0)
    ax2.set_title("Correlation heatmap")
    st.pyplot(fig2)

st.sidebar.header("파일 다운로드")
# provide download of current dataframe and correlations
buf = BytesIO()
df.to_csv(buf, index=False)
buf.seek(0)
st.sidebar.download_button("CSV로 데이터 다운로드", data=buf, file_name="fitness_data.csv", mime="text/csv")
'''

requirements = """streamlit
pandas
numpy
matplotlib
seaborn
openpyxl
"""

# write files
app_path = os.path.join(out_dir, "app.py")
with open(app_path, "w", encoding="utf-8") as f:
    f.write(app_py)

req_path = os.path.join(out_dir, "requirements.txt")
with open(req_path, "w", encoding="utf-8") as f:
    f.write(requirements)

# Also copy the uploaded Excel next to the app so GitHub repo will include it if desired
try:
    df.to_excel(os.path.join(out_dir, "fitness data.xlsx"), index=False)
except Exception:
    pass

# 5) Output a short summary to the user
print(f"Created Streamlit app at: {app_path}")
print(f"Created requirements.txt at: {req_path}")

print("\\nDetected body-fat column candidate:", bodyfat_col)
if top_corr_col is not None:
    print(f"Attribute most correlated with '{bodyfat_col}': '{top_corr_col}' with correlation {top_corr_val:.3f}")
else:
    print("Couldn't determine top correlated attribute (not enough numeric data).")

# List created files for download
for filename in os.listdir(out_dir):
    print("-", filename)

# Provide paths for the user to download in the chat response.
output_files = {
    "app.py": app_path,
    "requirements.txt": req_path,
    "bundled_excel": os.path.join(out_dir, "fitness data.xlsx")
}
output_files

