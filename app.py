"""Streamlit interactive dashboard for the AI-Based Student Performance Prediction System."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import load_artifacts, predict_batch, predict_single
from src.preprocess import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_OPTIONS,
    NUMERIC_BOUNDS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_dataset,
)
from src.train import train_and_evaluate

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="AI Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #f8fafc;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        color: #ffffff;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_loaded_artifacts():
    return load_artifacts(MODELS_DIR)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_dataset()


def ensure_models_ready() -> bool:
    required = [
        MODELS_DIR / "best_model.joblib",
        MODELS_DIR / "all_models.joblib",
        MODELS_DIR / "metrics.json",
    ]
    if all(path.exists() for path in required):
        return True

    st.warning("Trained model artifacts were not found yet.")
    if st.button("Train Models Now", type="primary"):
        with st.spinner("Training and benchmarking 7 ML models..."):
            train_and_evaluate(models_dir=MODELS_DIR)
        st.cache_resource.clear()
        st.rerun()
    return False


def render_model_comparison_tab(metrics: dict) -> None:
    st.subheader("Model Evaluation & Comparison Dashboard")
    leaderboard_df = pd.DataFrame(metrics["leaderboard"])
    best_row = leaderboard_df.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Performing Model", best_row["model"])
    col2.metric("Test R² Score", f"{best_row['test_r2']:.4f}")
    col3.metric("Test RMSE", f"{best_row['test_rmse']:.4f}")
    col4.metric(
        "5-Fold CV R²",
        f"{best_row['cv_r2_mean']:.4f} ± {best_row['cv_r2_std']:.4f}",
    )

    st.markdown("### Benchmark Leaderboard (Sorted by Lowest Test RMSE)")
    display_df = leaderboard_df.rename(
        columns={
            "model": "Model",
            "test_r2": "Test R²",
            "test_rmse": "Test RMSE",
            "test_mae": "Test MAE",
            "cv_r2_mean": "5-Fold CV R²",
            "cv_r2_std": "CV R² Std",
            "cv_rmse_mean": "CV RMSE",
            "cv_mae_mean": "CV MAE",
            "fit_time_sec": "Train Time (s)",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        r2_melted = leaderboard_df.melt(
            id_vars="model",
            value_vars=["test_r2", "cv_r2_mean"],
            var_name="Metric",
            value_name="R² Score",
        )
        r2_melted["Metric"] = r2_melted["Metric"].map(
            {"test_r2": "Test R²", "cv_r2_mean": "5-Fold CV R²"}
        )
        fig_r2 = px.bar(
            r2_melted,
            x="model",
            y="R² Score",
            color="Metric",
            barmode="group",
            title="R² Score Comparison Across Models (Higher is Better)",
            text_auto=".3f",
        )
        fig_r2.update_layout(xaxis_title="", yaxis_range=[0, 1.0], legend_title="")
        st.plotly_chart(fig_r2, use_container_width=True)

    with chart_col2:
        err_melted = leaderboard_df.melt(
            id_vars="model",
            value_vars=["test_rmse", "test_mae"],
            var_name="Metric",
            value_name="Error (Score Points)",
        )
        err_melted["Metric"] = err_melted["Metric"].map(
            {"test_rmse": "Test RMSE", "test_mae": "Test MAE"}
        )
        fig_err = px.bar(
            err_melted,
            x="model",
            y="Error (Score Points)",
            color="Metric",
            barmode="group",
            title="Prediction Error Comparison (Lower is Better)",
            text_auto=".2f",
        )
        fig_err.update_layout(xaxis_title="", legend_title="")
        st.plotly_chart(fig_err, use_container_width=True)

    st.divider()
    st.markdown("### Diagnostic Deep-Dive by Model")
    model_names = [row["model"] for row in metrics["leaderboard"]]
    selected_diag_model = st.selectbox(
        "Inspect Diagnostics & Feature Importance for Model:",
        options=model_names,
        index=0,
    )

    actual = metrics["test_actual"]
    preds = metrics["test_predictions"][selected_diag_model]
    diag_df = pd.DataFrame(
        {
            "Actual Exam Score": actual,
            "Predicted Exam Score": preds,
            "Residual (Actual - Predicted)": [a - p for a, p in zip(actual, preds)],
        }
    )

    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        fig_scatter = px.scatter(
            diag_df,
            x="Actual Exam Score",
            y="Predicted Exam Score",
            opacity=0.6,
            title=f"Actual vs. Predicted Scores ({selected_diag_model})",
        )
        min_val = min(min(actual), min(preds))
        max_val = max(max(actual), max(preds))
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Ideal Fit (y = x)",
                line=dict(color="#ef4444", dash="dash"),
            )
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with diag_col2:
        fig_res = px.histogram(
            diag_df,
            x="Residual (Actual - Predicted)",
            nbins=35,
            title=f"Residual Error Distribution ({selected_diag_model})",
            color_discrete_sequence=["#3b82f6"],
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="#ef4444")
        st.plotly_chart(fig_res, use_container_width=True)

    importances = metrics["feature_importance"].get(selected_diag_model, {})
    if importances:
        imp_df = pd.DataFrame(
            {
                "Feature": list(importances.keys()),
                "Relative Importance (%)": list(importances.values()),
            }
        ).head(12)
        fig_imp = px.bar(
            imp_df.sort_values("Relative Importance (%)"),
            x="Relative Importance (%)",
            y="Feature",
            orientation="h",
            title=f"Top 12 Influential Features ({selected_diag_model})",
            text_auto=".1f",
            color="Relative Importance (%)",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_imp, use_container_width=True)


def render_prediction_tab(
    all_models: dict,
    selected_model_name: str,
    cohort_mean: float,
) -> None:
    st.subheader("Single Student Performance Prediction & Risk Assessment")
    st.caption(
        "Adjust the student's academic habits, lifestyle, and environment factors below."
    )

    with st.form("student_prediction_form"):
        c1, c2, c3 = st.columns(3)
        student_input: dict = {}

        with c1:
            st.markdown("**Academic & Study Factors**")
            for feat in ["Hours_Studied", "Attendance", "Previous_Scores", "Tutoring_Sessions"]:
                low, high, default = NUMERIC_BOUNDS[feat]
                student_input[feat] = st.slider(
                    feat.replace("_", " "), min_value=low, max_value=high, value=default
                )
            for feat in ["Access_to_Resources", "Teacher_Quality", "School_Type"]:
                opts = CATEGORICAL_OPTIONS[feat]
                student_input[feat] = st.selectbox(
                    feat.replace("_", " "), options=opts, index=min(1, len(opts) - 1)
                )

        with c2:
            st.markdown("**Personal & Lifestyle Factors**")
            for feat in ["Sleep_Hours", "Physical_Activity"]:
                low, high, default = NUMERIC_BOUNDS[feat]
                student_input[feat] = st.slider(
                    feat.replace("_", " "), min_value=low, max_value=high, value=default
                )
            for feat in [
                "Motivation_Level",
                "Extracurricular_Activities",
                "Learning_Disabilities",
                "Peer_Influence",
                "Gender",
            ]:
                opts = CATEGORICAL_OPTIONS[feat]
                student_input[feat] = st.selectbox(
                    feat.replace("_", " "), options=opts, index=min(1, len(opts) - 1)
                )

        with c3:
            st.markdown("**Home & Background Factors**")
            for feat in [
                "Parental_Involvement",
                "Parental_Education_Level",
                "Family_Income",
                "Internet_Access",
                "Distance_from_Home",
            ]:
                opts = CATEGORICAL_OPTIONS[feat]
                student_input[feat] = st.selectbox(
                    feat.replace("_", " "), options=opts, index=min(1, len(opts) - 1)
                )

        submitted = st.form_submit_button("Predict Exam Score", type="primary", use_container_width=True)

    if submitted:
        pipeline = all_models[selected_model_name]
        result = predict_single(student_input, pipeline, cohort_mean=cohort_mean)
        risk = result["risk"]

        res_c1, res_c2, res_c3 = st.columns(3)
        res_c1.metric(
            f"Predicted Exam Score ({selected_model_name})",
            f"{result['predicted_score']:.2f} / 100",
        )
        res_c2.metric(
            "Difference vs. Cohort Average",
            f"{result['delta_vs_cohort']:+.2f} pts",
            help=f"Cohort mean score is {cohort_mean:.2f}",
        )
        with res_c3:
            st.markdown("**Academic Risk Tier**")
            st.markdown(
                f"<span class='risk-badge' style='background-color:{risk['color']}'>{risk['tier']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(risk["summary"])

        st.markdown("### Personalized Study & Support Recommendations")
        for tip in result["recommendations"]:
            st.info(tip)

        st.markdown("### Multi-Model Consensus for This Student")
        consensus_rows = []
        for m_name, m_pipe in all_models.items():
            m_res = predict_single(student_input, m_pipe, cohort_mean=cohort_mean)
            consensus_rows.append(
                {
                    "Model": m_name,
                    "Predicted Score": m_res["predicted_score"],
                    "Risk Tier": m_res["risk"]["tier"],
                }
            )
        st.dataframe(pd.DataFrame(consensus_rows), use_container_width=True, hide_index=True)


def render_eda_tab(df: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Exploratory Data Analysis (Kaggle Student Performance Factors)")
    ds = metrics["dataset_summary"]
    t_stats = ds["target_stats"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{ds['row_count']:,}")
    m2.metric("Input Features", f"{ds['feature_count']} (6 Num / 13 Cat)")
    m3.metric("Mean Exam Score", f"{t_stats['mean']:.2f}")
    m4.metric("Score Range", f"{t_stats['min']:.0f} – {t_stats['max']:.0f}")

    if ds["missing_counts"]:
        missing_str = ", ".join(f"{k}: {v}" for k, v in ds["missing_counts"].items())
        st.caption(
            f"Missing values detected and automatically imputed in pipeline → {missing_str}"
        )

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            df,
            x=TARGET_COLUMN,
            nbins=35,
            title="Distribution of Student Exam Scores",
            color_discrete_sequence=["#6366f1"],
        )
        fig_hist.add_vline(
            x=t_stats["mean"],
            line_dash="dash",
            line_color="#10b981",
            annotation_text=f"Mean ({t_stats['mean']})",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        corr_df = df[NUMERIC_FEATURES + [TARGET_COLUMN]].corr(numeric_only=True).round(2)
        fig_corr = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Numeric Feature Correlation Heatmap",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Feature Relationship Explorer")
    selected_feature = st.selectbox(
        "Select a feature to inspect against Exam Score:",
        options=ALL_FEATURES,
        index=0,
    )
    if selected_feature in NUMERIC_FEATURES:
        fig_rel = px.scatter(
            df,
            x=selected_feature,
            y=TARGET_COLUMN,
            opacity=0.45,
            trendline=None,
            title=f"{selected_feature.replace('_', ' ')} vs. Exam Score",
            color_discrete_sequence=["#0ea5e9"],
        )
    else:
        fig_rel = px.box(
            df,
            x=selected_feature,
            y=TARGET_COLUMN,
            color=selected_feature,
            title=f"Exam Score Distribution by {selected_feature.replace('_', ' ')}",
        )
    st.plotly_chart(fig_rel, use_container_width=True)


def render_batch_tab(df: pd.DataFrame, all_models: dict, selected_model_name: str) -> None:
    st.subheader("Batch CSV Student Performance Prediction")
    st.write(
        "Upload a CSV containing the 19 student feature columns to generate predictions and risk classifications in bulk."
    )

    sample_template = df[ALL_FEATURES].head(15).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Sample Input CSV Template (15 rows)",
        data=sample_template,
        file_name="sample_students_input.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Student Records CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            pipeline = all_models[selected_model_name]
            scored_df = predict_batch(batch_df, pipeline)

            st.success(
                f"Generated predictions for {len(scored_df)} students using {selected_model_name}."
            )
            tier_counts = (
                scored_df["Risk_Tier"].value_counts().rename_axis("Risk Tier").reset_index(name="Students")
            )
            c1, c2 = st.columns([1, 2])
            with c1:
                fig_pie = px.pie(
                    tier_counts,
                    names="Risk Tier",
                    values="Students",
                    title="Predicted Risk Tier Breakdown",
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                st.dataframe(scored_df.head(50), use_container_width=True, hide_index=True)

            out_csv = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=out_csv,
                file_name="predicted_student_performance.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as exc:
            st.error(f"Could not process uploaded CSV: {exc}")


def main() -> None:
    st.title("🎓 AI-Based Student Performance Prediction System")
    st.caption(
        "Final Year Academic Project • Kaggle Student Performance Factors Dataset • Multi-Model Regression & Risk Assessment"
    )

    if not ensure_models_ready():
        return

    best_model, all_models, metrics = get_loaded_artifacts()
    df = get_dataset()

    with st.sidebar:
        st.header("System Controls")
        model_options = [row["model"] for row in metrics["leaderboard"]]
        selected_model = st.selectbox(
            "Active Prediction Model",
            options=model_options,
            index=0,
            help="Defaults to the best-performing model on test RMSE.",
        )

        st.divider()
        st.markdown("**Risk Tier Thresholds**")
        st.markdown(
            "- **High Risk:** `< 64.0`\n"
            "- **Moderate Risk:** `64.0 – 67.4`\n"
            "- **On Track:** `67.5 – 71.9`\n"
            "- **High Performer:** `>= 72.0`"
        )

        st.divider()
        if st.button("Retrain All Models", use_container_width=True):
            with st.spinner("Retraining models and updating metrics..."):
                train_and_evaluate(models_dir=MODELS_DIR)
            st.cache_resource.clear()
            st.rerun()

    tab_compare, tab_predict, tab_eda, tab_batch = st.tabs(
        [
            "📊 Model Comparison Dashboard",
            "🎯 Student Prediction",
            "📈 Dataset & EDA Insights",
            "📁 Batch CSV Prediction",
        ]
    )

    cohort_mean = metrics["dataset_summary"]["target_stats"]["mean"]

    with tab_compare:
        render_model_comparison_tab(metrics)
    with tab_predict:
        render_prediction_tab(all_models, selected_model, cohort_mean)
    with tab_eda:
        render_eda_tab(df, metrics)
    with tab_batch:
        render_batch_tab(df, all_models, selected_model)


if __name__ == "__main__":
    main()
