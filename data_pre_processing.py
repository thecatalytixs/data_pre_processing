import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, LabelEncoder
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Halal Data Transformation Viewer", layout="wide")
st.title("Halal Authentication Data - Transformation Visualizer")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your FTIR dataset (CSV format only)", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    else:
        return None

df = load_data(uploaded_file)

if df is not None:
    st.subheader("1. Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Drop non-numeric for transformation
    non_numeric_cols = ["SampleID", "Class"]
    features = df.drop(columns=[col for col in non_numeric_cols if col in df.columns], errors='ignore')

    # Transformation functions
    def standardize_n(data): return (data - data.mean()) / data.std(ddof=0)
    def center(data): return data - data.mean()
    def std_n1(data): return data / data.std(ddof=1)
    def std_n(data): return data / data.std(ddof=0)
    def rescale_0_1(data): return pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)
    def rescale_0_100(data): return ((data - data.min()) / (data.max() - data.min())) * 100
    def pareto(data): return (data - data.mean()) / np.sqrt(data.std())
    def binarize(data): return (data > 0).astype(int)
    def sign_func(data): return np.sign(data)
    def arcsin_func(data): return np.arcsin(np.sqrt(np.clip(data, 0, 1)))
    def boxcox_func(data):
        df_out = pd.DataFrame()
        for col in data.columns:
            positive = data[col] + abs(data[col].min()) + 1
            df_out[col], _ = stats.boxcox(positive)
        return df_out
    def winsorize_func(data): return pd.DataFrame({col: stats.mstats.winsorize(data[col], limits=[0.05, 0.05]) for col in data.columns})
    def johnson_func(data): return pd.DataFrame(PowerTransformer(method='yeo-johnson').fit_transform(data), columns=data.columns)

    transformations = {
        "Standardize (n)": standardize_n,
        "Center": center,
        "/ Std dev (n-1)": std_n1,
        "/ Std dev (n)": std_n,
        "Rescale 0–1": rescale_0_1,
        "Rescale 0–100": rescale_0_100,
        "Pareto scaling": pareto,
        "Binarize (0/1)": binarize,
        "Sign (-1/0/1)": sign_func,
        "Arcsin": arcsin_func,
        "Box-Cox": boxcox_func,
        "Winsorize": winsorize_func,
        "Johnson": johnson_func
    }

    st.subheader("2. Select Transformations to Compare")
    selected = st.multiselect("Choose transformations", list(transformations.keys()))

    if selected:
        for name in selected:
            st.markdown(f"### 🔄 {name}")
            try:
                transformed = transformations[name](features.copy())
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                sns.boxplot(data=features, ax=axes[0])
                axes[0].set_title("Original Data")
                sns.boxplot(data=transformed, ax=axes[1])
                axes[1].set_title(f"After {name}")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"⚠️ Transformation '{name}' failed: {e}")
    else:
        st.info("Please select at least one transformation method to display the boxplot comparisons.")
else:
    st.info("Please upload your dataset to begin.")
