"""
Step 3 — Model Benchmarking (Premium UI Edition)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.model_trainer import MODEL_REGISTRY, train_model, get_predictions
from utils.metrics import compute_metrics

RANDOM_SEED = 42
ALL_MODELS  = list(MODEL_REGISTRY.keys())
PALETTE     = ["#06d6a0", "#7c6fe0", "#ffb703", "#f97316", "#60a5fa"]

DARK_CHART = dict(
    paper_bgcolor="#101622", plot_bgcolor="#101622",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=60, r=30, t=60, b=50),
)


def _radar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    categories = ["Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "F1_Score", "ROC_AUC"]
    fig = go.Figure()
    
    def hex_to_rgba(h, alpha=0.15):
        h = h.lstrip('#')
        return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"

    for i, row in metrics_df.iterrows():
        vals = [row.get(c, 0) or 0 for c in categories]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["Model"],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            fillcolor=hex_to_rgba(PALETTE[i % len(PALETTE)], 0.15),
        ))
    fig.update_layout(
        **DARK_CHART,
        polar=dict(
            radialaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.06)", color="#7a8fa6"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="#7a8fa6"),
            bgcolor="#101622",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        title="Model Radar Chart",
        height=380,
    )
    return fig


def _bar_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    metric_cols = ["Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "F1_Score", "ROC_AUC"]
    metric_names = ["Accuracy", "Bal. Accuracy", "Sensitivity", "Specificity", "F1", "ROC-AUC"]
    bar_palette  = ["#06d6a0", "#7c6fe0", "#ffb703", "#f97316", "#60a5fa", "#fb7185"]
    fig = go.Figure()
    for i, (col, label) in enumerate(zip(metric_cols, metric_names)):
        if col not in metrics_df.columns: continue
        fig.add_trace(go.Bar(
            name=label,
            x=metrics_df["Model"].tolist(),
            y=metrics_df[col].tolist(),
            marker_color=bar_palette[i % len(bar_palette)],
            marker_line=dict(color="#101622", width=1),
        ))
    fig.update_layout(
        **DARK_CHART,
        barmode="group",
        title="Metric Comparison by Model",
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        xaxis=dict(color="#7a8fa6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )
    return fig


def render():
    from utils.ui_components import section_header, icon

    if not st.session_state.get("step2_complete"):
        st.warning("Complete **Step 2 — Feature Engineering** first.")
        return

    X_train = st.session_state["X_train"]
    X_test  = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test  = st.session_state["y_test"]

    st.markdown(f"""<p>Training on <strong style="color:var(--teal);">{len(X_train):,} samples</strong>
    · {X_train.shape[1]} features · evaluating on
    <strong style="color:var(--amber);">{len(X_test):,} test samples</strong>.</p>""",
    unsafe_allow_html=True)

    # Model checkboxes
    section_header("trophy", "Select Models to Benchmark")
    cols = st.columns(len(ALL_MODELS))
    selected_models = [name for i, name in enumerate(ALL_MODELS)
                       if cols[i].checkbox(name, value=True, key=f"bm_{name}")]
    if not selected_models:
        st.error("Select at least one model.")
        return

    # Options
    section_header("settings", "Training Options")
    with st.expander("Tuning configuration", expanded=False):
        c1, c2, c3 = st.columns(3)
        tune    = c1.checkbox("Hyperparameter tuning (RandomizedSearchCV)", value=False, help="Uncheck to train instantly without tuning.")
        n_iter  = c2.slider("Search iterations", 5, 50, 5, 5, disabled=not tune, help="Lower this if SVM is taking too long.")
        cv_folds= c3.slider("CV folds", 3, 10, 3, 1, disabled=not tune, help="Lower folds to speed up cross-validation.")

    if st.button("Run Benchmarking", type="primary", use_container_width=True):
        results_list   = []
        trained_models = {}

        with st.status(f"Benchmarking {len(selected_models)} models...", expanded=True) as status:
            for idx, model_name in enumerate(selected_models):
                status.write(f"⏳ Training **{model_name}**...")
                try:
                    model, best_params, cv_res = train_model(
                        model_name, X_train, y_train, tune=tune, n_iter=n_iter, cv_folds=cv_folds)
                    y_pred, y_proba = get_predictions(model, X_test)
                    metrics = compute_metrics(y_test.values, y_pred, y_proba, model_name=model_name)
                    metrics["CV_AUC"] = cv_res.get("best_cv_auc")
                    results_list.append(metrics)
                    trained_models[model_name] = {
                        "model": model, "best_params": best_params,
                        "cv_results": cv_res, "y_pred": y_pred, "y_proba": y_proba,
                    }
                    status.write(f"✅ {model_name} complete — ROC-AUC: {metrics.get('ROC_AUC','—')}")
                except Exception as e:
                    status.write(f"❌ {model_name} failed: {e}")
            
            status.update(label="Benchmarking complete!", state="complete", expanded=False)
            
        st.session_state.update({
            "benchmark_results": results_list, "trained_models": trained_models,
            "step3_complete": True,
        })
        st.success("Benchmarking complete. Select your model below and switch to **Final Model**.")

    if not st.session_state.get("step3_complete"):
        return

    results_list   = st.session_state["benchmark_results"]
    trained_models = st.session_state.get("trained_models", {})
    metrics_df     = pd.DataFrame([r for r in results_list if "Error" not in r])
    if metrics_df.empty:
        st.error("All models failed. Check your data.")
        return

    leaderboard = metrics_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    leaderboard.index += 1

    # Leaderboard
    section_header("trophy", "Leaderboard", "Ranked by ROC-AUC on the test set",
                   "var(--amber-dim)", "#ffb703")

    # Medal indicators
    medal_html = '<div style="display:flex;gap:0.75rem;margin-bottom:1rem;flex-wrap:wrap;">'
    medals = ["#FFD700", "#C0C0C0", "#CD7F32"]
    for i, row in leaderboard.head(3).iterrows():
        clr = medals[i-1]
        medal_html += f"""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
             padding:0.85rem 1.2rem;display:flex;align-items:center;gap:0.75rem;flex:1;min-width:180px;">
          <div style="width:32px;height:32px;border-radius:50%;background:rgba(255,255,255,0.05);
               display:flex;align-items:center;justify-content:center;font-size:1rem;color:{clr};font-weight:800;">#{i}</div>
          <div>
            <div style="font-size:0.85rem;font-weight:700;color:var(--text);">{row['Model']}</div>
            <div style="font-size:0.72rem;color:{clr};font-weight:600;">ROC-AUC: {row.get('ROC_AUC','—') if row.get('ROC_AUC') is None else f"{row.get('ROC_AUC'):.4f}"}</div>
          </div>
        </div>"""
    medal_html += "</div>"
    st.markdown(medal_html, unsafe_allow_html=True)

    # Format floats for display to avoid pandas Styler rendering bugs
    display_df = leaderboard.copy()
    float_cols = ["Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "F1_Score", "ROC_AUC", "CV_AUC"]
    for c in float_cols:
        if c in display_df.columns:
            display_df[c] = display_df[c].apply(lambda x: f"{x:.4f}" if pd.notnull(x) and isinstance(x, (int, float)) else x)
            
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Charts — side by side
    section_header("chart", "Visual Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_bar_comparison(metrics_df), use_container_width=True)
    with col2:
        st.plotly_chart(_radar_chart(metrics_df), use_container_width=True)

    # Best hyperparams
    with st.expander("Best Hyperparameters per Model"):
        for name, info in trained_models.items():
            if info.get("best_params"):
                st.markdown(f"**{name}**")
                st.json(info["best_params"])

    # Model selection
    section_header("target", "Select Model for Final Training")
    best_model = leaderboard.iloc[0]["Model"] if not leaderboard.empty else selected_models[0]
    model_list = metrics_df["Model"].tolist()
    chosen = st.radio(
        "Preferred model:",
        options=model_list,
        index=model_list.index(best_model) if best_model in model_list else 0,
        horizontal=True,
        key="selected_model_name",
        label_visibility="collapsed",
    )
    st.session_state["chosen_model_name"] = chosen
    st.session_state["chosen_model"] = trained_models.get(chosen, {}).get("model")
    st.success(f"Selected **{chosen}**. Switch to the **Final Model** tab.")
