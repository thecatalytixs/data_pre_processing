import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, Binarizer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io
from math import pi

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

    transformed_dfs = {}
    scores = []

    for method in selected:
        df_copy = numeric_df.copy()
        try:
            if method == "Standardize (n)":
                df_t = StandardScaler().fit_transform(df_copy)
            elif method == "Center":
                df_t = df_copy - df_copy.mean()
            elif method == "Std dev (n-1)":
                df_t = df_copy / df_copy.std(ddof=1)
            elif method == "Std dev (n)":
                df_t = df_copy / df_copy.std(ddof=0)
            elif method == "Rescale 0-1":
                df_t = MinMaxScaler().fit_transform(df_copy)
            elif method == "Rescale 0-100":
                df_t = MinMaxScaler(feature_range=(0, 100)).fit_transform(df_copy)
            elif method == "Pareto scaling":
                df_t = (df_copy - df_copy.mean()) / np.sqrt(df_copy.std())
            elif method == "Binarize (0/1)":
                df_t = Binarizer().fit_transform(df_copy)
            elif method == "Sign (-1/0/1)":
                df_t = np.sign(df_copy)
            elif method == "Arcsin":
                df_t = np.arcsin(df_copy.clip(0, 1))
            elif method == "Box-Cox":
                df_t = df_copy.copy()
                for col in df_t.columns:
                    min_val = df_t[col].min()
                    df_t[col] = df_t[col] + 1 - min_val if min_val <= 0 else df_t[col]
                    df_t[col], _ = stats.boxcox(df_t[col])
            elif method == "Winsorize":
                df_t = df_copy.copy()
                for col in df_t.columns:
                    df_t[col] = stats.mstats.winsorize(df_t[col], limits=[0.05, 0.05])
            elif method == "Johnson":
                df_t = df_copy.copy()
                for col in df_t.columns:
                    fitted_data, _, _, _ = stats.johnsonsu.fit(df_t[col])
                    df_t[col] = stats.johnsonsu.pdf(df_t[col], *stats.johnsonsu.fit(df_t[col]))
            else:
                df_t = df_copy

            df_trans = pd.DataFrame(df_t, columns=df_copy.columns)
            transformed_dfs[method] = df_trans

            skewness = np.abs(df_trans.skew()).mean()
            outliers = ((df_trans - df_trans.mean()).abs() > 3 * df_trans.std()).sum().sum()
            iqr = (df_trans.quantile(0.75) - df_trans.quantile(0.25)).mean()
            normality = np.mean([stats.shapiro(df_trans[col])[1] for col in df_trans.columns if len(df_trans[col].dropna()) > 3])

            scores.append({
                "name": method,
                "skewness_score": 1 - min(skewness / 10, 1),
                "outliers_score": 1 - min(outliers / (len(df_trans) * len(df_trans.columns)), 1),
                "iqr_score": 1 - min(iqr / 10, 1),
                "normality_score": normality,
                "total_score": (1 - min(skewness / 10, 1) + 1 - min(outliers / (len(df_trans) * len(df_trans.columns)), 1) + 1 - min(iqr / 10, 1) + normality) / 4
            })

            # Display Boxplots
            st.markdown(f"## 🔁 {method}")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.boxplot(data=numeric_df, ax=axes[0])
            axes[0].set_title("Original Data")
            sns.boxplot(data=df_trans, ax=axes[1])
            axes[1].set_title(f"After {method}")
            st.pyplot(fig)

            # Download button for transformed data (top right corner)
            csv_data = df_trans.to_csv(index=False).encode('utf-8')
            col1, col2 = st.columns([0.85, 0.15])
            with col2:
                st.download_button(
                    label=f"📥 Download CSV",
                    data=csv_data,
                    file_name=f"{method.lower().replace(' ', '_')}_transformed.csv",
                    mime='text/csv',
                    key=method
                )

        except Exception as e:
            st.warning(f"⚠️ Transformation {method} failed: {e}")

    # -----------------------------
    # 4. Display Score Table + Export CSV
    # -----------------------------
    if scores:
        df_scores = pd.DataFrame(scores).sort_values(by="total_score", ascending=False)
        st.markdown("### 📊 Transformation Score Table")
        st.dataframe(df_scores, use_container_width=True)

        csv = df_scores.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download Score Table as CSV", data=csv, file_name="transformation_scores.csv", mime='text/csv')

        # -----------------------------
        # 5. Matplotlib Bar Chart + PNG Export
        # -----------------------------
        st.subheader("Bar Chart of Transformation Scores (Matplotlib)")
        fig_bar, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df_scores['name'], df_scores['total_score'], color='skyblue')
        ax.set_xlabel('Transformation Method')
        ax.set_ylabel('Total Score')
        ax.set_title('Total Scores by Transformation')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig_bar)

        buf = io.BytesIO()
        fig_bar.savefig(buf, format="png", bbox_inches="tight")
        st.download_button("🖼️ Download Bar Chart as PNG", data=buf.getvalue(), file_name="bar_chart_transformation_scores.png", mime="image/png")

        # -----------------------------
        # 6. Matplotlib Radar Chart + PNG Export
        # -----------------------------
        st.subheader("Radar Chart of Top 5 Transformations (Matplotlib)")
        radar_df = df_scores.head(5)
        metrics = ["skewness_score", "outliers_score", "iqr_score", "normality_score"]
        labels = ["Skewness", "Outliers", "IQR", "Normality"]
        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        fig_radar, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        colors = plt.cm.tab10.colors

        for i, (_, row) in enumerate(radar_df.iterrows()):
            values = [row[m] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, label=row["name"], color=colors[i % len(colors)], linewidth=2)
            ax.fill(angles, values, alpha=0.2, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        ax.set_title("Radar Chart of Metric Profiles (Normalized)", size=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
        st.pyplot(fig_radar)

        radar_buf = io.BytesIO()
        fig_radar.savefig(radar_buf, format='png', bbox_inches='tight')
        st.download_button("🕸️ Download Radar Chart as PNG", data=radar_buf.getvalue(), file_name="radar_chart_metric_profiles.png", mime="image/png")
