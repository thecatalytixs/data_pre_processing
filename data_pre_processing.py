import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, LabelEncoder
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import skew, iqr, shapiro

# --- PAGE SETUP ---
st.set_page_config(page_title="Halal Authentication - Data Transformation Viewer", layout="wide")
st.title("Halal Authentication Data - Transformation Visualizer & Evaluator")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your FTIR dataset (CSV format only)", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file) if file else None

df = load_data(uploaded_file)

if df is not None:
    st.subheader("1. Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --- FEATURE SELECTION ---
    exclude_cols = ["SampleID", "Class"]
    features = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')

    # --- TRANSFORMATION FUNCTIONS ---
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

    # --- TRANSFORMATION SELECTION ---
    st.subheader("2. Select Transformations to Compare")
    selected = st.multiselect("Choose transformations", list(transformations.keys()))

    # --- BOX PLOTS ---
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
                st.warning(f"{name} failed: {e}")

        # --- SCORING INTERPRETATION ---
        st.subheader("3. Interpretation: Best Transformation")

        scores = []
        for name in selected:
            try:
                transformed = transformations[name](features.copy())
                skews = transformed.apply(lambda x: skew(x, nan_policy='omit'))
                total_skew = np.sum(np.abs(skews))

                outlier_count = 0
                for col in transformed.columns:
                    Q1 = transformed[col].quantile(0.25)
                    Q3 = transformed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outliers = ((transformed[col] < lower) | (transformed[col] > upper)).sum()
                    outlier_count += outliers

                total_iqr = transformed.apply(iqr).sum()

                shapiro_pvals = []
                for col in transformed.columns:
                    try:
                        _, p = shapiro(transformed[col].sample(min(5000, len(transformed[col]))))
                        shapiro_pvals.append(p)
                    except:
                        pass
                avg_pval = np.mean(shapiro_pvals) if shapiro_pvals else 0

                scores.append({
                    "name": name,
                    "skewness": total_skew,
                    "outliers": outlier_count,
                    "iqr": total_iqr,
                    "normality": avg_pval
                })

            except Exception:
                continue

        if scores:
            df_scores = pd.DataFrame(scores)
            for col in ["skewness", "outliers", "iqr"]:
                df_scores[col + "_score"] = 1 - (df_scores[col] - df_scores[col].min()) / (df_scores[col].max() - df_scores[col].min() + 1e-10)
            df_scores["normality_score"] = df_scores["normality"] / (df_scores["normality"].max() + 1e-10)

            df_scores["final_score"] = (
                0.4 * df_scores["skewness_score"] +
                0.3 * df_scores["outliers_score"] +
                0.2 * df_scores["iqr_score"] +
                0.1 * df_scores["normality_score"]
            )

            df_scores = df_scores.sort_values(by="final_score", ascending=False)
            best_transform = df_scores.iloc[0]["name"]

            st.markdown(f"### 🏆 Best Transformation: **{best_transform}**")
            st.dataframe(df_scores[["name", "skewness", "outliers", "iqr", "normality", "final_score"]].round(3), use_container_width=True)

            # --- VISUALIZATIONS ---
            st.subheader("4. Visual Comparison of Scores")

            # Bar chart
            fig_bar = px.bar(
                df_scores,
                x="name",
                y="final_score",
                color="final_score",
                title="Final Composite Score by Transformation",
                labels={"name": "Transformation", "final_score": "Score"},
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(xaxis_title="Transformation", yaxis_title="Composite Score")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Radar chart
            st.markdown("### 🕸️ Radar Chart of Top 5 Transformations")
            radar_df = df_scores.head(5)
            metrics = ["skewness_score", "outliers_score", "iqr_score", "normality_score"]
            categories = ["Skewness", "Outliers", "IQR", "Normality"]

            fig_radar = go.Figure()
            for _, row in radar_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=categories,
                    fill='toself',
                    name=row["name"]
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Radar Chart of Metric Profiles (Normalized)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # --- DOWNLOAD SECTION ---
            st.subheader("5. Download Results")

            csv_data = df_scores.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Score Table as CSV",
                data=csv_data,
                file_name="transformation_scores.csv",
                mime="text/csv"
            )

            try:
                bar_buf = pio.to_image(fig_bar, format="png")
                st.download_button(
                    label="🖼️ Download Bar Chart as PNG",
                    data=bar_buf,
                    file_name="bar_chart_transformation_scores.png",
                    mime="image/png"
                )
            except Exception as e:
                st.warning(f"Bar chart export failed. Try `pip install kaleido`: {e}")

            try:
                radar_buf = pio.to_image(fig_radar, format="png")
                st.download_button(
                    label="🖼️ Download Radar Chart as PNG",
                    data=radar_buf,
                    file_name="radar_chart_metric_profiles.png",
                    mime="image/png"
                )
            except Exception as e:
                st.warning(f"Radar chart export failed. Try `pip install kaleido`: {e}")

        else:
            st.warning("No transformation could be processed successfully.")

else:
    st.info("Please upload a CSV file to begin.")
