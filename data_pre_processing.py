import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, Binarizer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# -----------------------------
# 1. Page setup
# -----------------------------
st.set_page_config(page_title="Halal Data Preprocessing App", layout="wide")
st.title("🧪 Halal Data Preprocessing App")

# -----------------------------
# 2. Upload dataset
# -----------------------------
st.markdown("### Step 1: Upload your dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_df = df.select_dtypes(include=[np.number])
    st.write("Preview of dataset:")
    st.dataframe(numeric_df.head(), use_container_width=True)

    # -----------------------------
    # KMO Test
    # -----------------------------
    st.markdown("### ✅ Dataset Adequacy Check (KMO and Bartlett’s Tests)")
    kmo_all, kmo_model = calculate_kmo(numeric_df.dropna())
    st.write(f"KMO Score: {kmo_model:.4f}")
    if kmo_model < 0.5:
        st.error("Dataset is **inadequate** for halal authentication (KMO < 0.5). Consider improving data quality.")
    elif kmo_model < 0.7:
        st.warning("Dataset is **mediocre** (0.5 < KMO < 0.7), but acceptable for halal authentication.")
    elif kmo_model < 0.8:
        st.success("Dataset is **good** (0.7 < KMO < 0.8) for halal authentication.")
    elif kmo_model < 0.9:
        st.success("Dataset is **very good** (0.8 < KMO < 0.9) for halal authentication.")
    else:
        st.success("Dataset is **excellent** (KMO > 0.9) for halal authentication.")

    # Bartlett’s Test of Sphericity
    chi_square_value, p_value = calculate_bartlett_sphericity(numeric_df.dropna())
    st.write(f"Bartlett’s Test: Chi-square = {chi_square_value:.2f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        st.success("Bartlett's test is significant (p < 0.05), indicating variables are correlated and the dataset is suitable for MDA in halal authentication.")
    else:
        st.error("Bartlett's test is **not** significant (p ≥ 0.05), suggesting variables may not be correlated enough for MDA.")

    # -----------------------------
    # 3. Choose transformations
    # -----------------------------
    st.markdown("### Step 2: Select Data Transformations")
    transformations = [
        "Standardize (n)", "Center", "Std dev (n-1)", "Std dev (n)", 
        "Rescale 0-1", "Rescale 0-100", "Pareto scaling", "Binarize (0/1)", 
        "Sign (-1/0/1)", "Arcsin", "Box-Cox", "Winsorize", "Johnson"
    ]

    all_selected = st.checkbox("Select All Transformations")
    if all_selected:
        selected = transformations
    else:
        selected = st.multiselect("Choose one or more transformations to apply", transformations)

    with st.expander("📘 Theory & Formula of Each Transformation"):
        st.markdown("""
        **Standardize (n)**: $$X_{scaled} = \frac{X - \mu}{\sigma}$$
        
        **Center**: $$X_{centered} = X - \mu$$

        **Std dev (n-1)**: $$X_{scaled} = \frac{X}{s}, \text{ where } s = \sqrt{\frac{1}{n-1}\sum (x_i - \bar{x})^2}$$

        **Std dev (n)**: $$X_{scaled} = \frac{X}{s}, \text{ where } s = \sqrt{\frac{1}{n}\sum (x_i - \bar{x})^2}$$

        **Rescale 0-1**: $$X' = \frac{X - X_{min}}{X_{max} - X_{min}}$$

        **Rescale 0-100**: $$X' = \frac{X - X_{min}}{X_{max} - X_{min}} \times 100$$

        **Pareto Scaling**: $$X' = \frac{X - \mu}{\sqrt{\sigma}}$$

        **Binarize (0/1)**: Set values above a threshold to 1, below or equal to 0.

        **Sign (-1/0/1)**: Assigns -1 for negative, 0 for zero, 1 for positive values.

        **Arcsin**: $$X' = \arcsin(\sqrt{X})$$ (commonly used for proportion data between 0 and 1)

        **Box-Cox**: $$X' = \frac{X^\lambda - 1}{\lambda}$$ if \( \lambda \neq 0 \); $$\log(X)$$ if \( \lambda = 0 \)

        **Winsorize**: Replaces extreme values by the nearest value within a specified percentile range.

        **Johnson**: Fits data to Johnson SU distribution and transforms it to resemble normal distribution.
        """)

    transformed_dfs = {}
    scores = []

    for method in selected:
        ... # Code continues unchanged
