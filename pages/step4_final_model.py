"""
Step 4 — Final Model Training & Evaluation (Premium UI Edition)
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.model_trainer import train_model, get_predictions
from utils.metrics import compute_metrics
from utils.descriptor_generator import generate_all_features
from utils.exporter import export_model_zip

RANDOM_SEED = 42
DARK_CHART = dict(
    paper_bgcolor="#101622", plot_bgcolor="#101622",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=60, r=30, t=60, b=50),
)
PALETTE = ["#06d6a0", "#7c6fe0", "#ffb703", "#f97316", "#60a5fa", "#fb7185"]


def _cm_chart(y_true, y_pred, title):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = ["Inactive (0)", "Active (1)"]
    cs = [[0, "#1a2540"], [0.5, "#0e5c4a"], [1, "#06d6a0"]]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=cs, showscale=False,
        text=cm, texttemplate="<b>%{text}</b>",
        textfont=dict(size=22, color="#e2e8f0"),
    ))
    fig.update_layout(**DARK_CHART, title=title, xaxis_title="Predicted", yaxis_title="Actual", height=300)
    return fig


def _roc_chart(y_true, y_proba, title):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}",
                             line=dict(color="#06d6a0", width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                             line=dict(color="#3d4e63", dash="dash", width=1.5)))
    fig.update_layout(**DARK_CHART, title=title, xaxis_title="FPR", yaxis_title="TPR",
                      legend=dict(x=0.6, y=0.1), height=300,
                      xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                      yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
    return fig


def _try_shap(model, X_train, model_name):
    try:
        import shap, matplotlib.pyplot as plt
        X_s = X_train.fillna(X_train.median(numeric_only=True)).head(200)
        tree_models = ["Random Forest", "LightGBM", "XGBoost"]
        if model_name not in tree_models:
            st.info("SHAP TreeExplainer is available for Random Forest, LightGBM, and XGBoost only.")
            return
            
        est = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(X_s)
        
        if isinstance(sv, list): 
            sv = sv[1]
        elif len(getattr(sv, "shape", [])) == 3:
            sv = sv[:, :, 1]
            
        with plt.style.context("dark_background"):
            shap.summary_plot(sv, X_s, plot_type="bar", show=False, max_display=20, color="#06d6a0")
            fig = plt.gcf()
            fig.patch.set_facecolor("#101622")
            ax = plt.gca()
            ax.set_facecolor("#101622")
            
            ax.tick_params(colors="#e2e8f0")
            ax.xaxis.label.set_color("#e2e8f0")
            
            # Subtle gridlines and spines
            ax.xaxis.grid(color="rgba(255,255,255,0.06)", linestyle="-", linewidth=1)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
                
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.clf()
            
    except Exception as e:
        st.info(f"SHAP plot unavailable: {e}")


def render():
    from utils.ui_components import section_header, icon

    if not st.session_state.get("step3_complete"):
        st.warning("Complete **Step 3 — Benchmarking** first.")
        return

    chosen_name  = st.session_state.get("chosen_model_name", "")
    chosen_model = st.session_state.get("chosen_model")
    X_train      = st.session_state["X_train"]
    X_test       = st.session_state["X_test"]
    y_train      = st.session_state["y_train"]
    y_test       = st.session_state["y_test"]
    feature_cols = st.session_state["feature_columns"]

    if not chosen_name or chosen_model is None:
        st.error("No model found in session. Go back to Benchmark and select a model.")
        return

    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:10px;background:var(--teal-dim);
         border:1px solid rgba(6,214,160,0.3);border-radius:99px;padding:6px 16px;margin-bottom:1rem;">
      {icon("target", 16, "#06d6a0")}
      <span style="font-size:0.84rem;font-weight:700;color:var(--teal);">Selected model: {chosen_name}</span>
    </div>""", unsafe_allow_html=True)

    # Options
    section_header("settings", "Training Options")
    with st.expander("Configure", expanded=False):
        retrain = st.checkbox("Re-train model from scratch on full train set", value=False,
                              help="Uses the already-tuned model from benchmarking by default.")
        tune_f  = st.checkbox("Tune during re-training", value=False, disabled=not retrain)
        n_it    = st.slider("Search iterations", 10, 100, 30, 10, disabled=not (retrain and tune_f))

    # Validation
    section_header("upload", "Optional Validation Evaluation", "Evaluate the final model on the held-out validation set")
    use_stored = st.checkbox("Use validation set from Step 1 (stored in session)", value=True)
    val_upload = st.file_uploader("Or upload a labelled validation CSV (SMILES + Activity)",
                                  type=["csv"], key="val_upload_s4")

    if st.button("Perform cross validation for selected model", type="primary", use_container_width=True):
        with st.spinner(f"Training {chosen_name}…"):
            if retrain:
                final_model, _, _ = train_model(chosen_name, X_train, y_train, tune=tune_f, n_iter=n_it)
            else:
                final_model = chosen_model

        with st.spinner("Evaluating on test set…"):
            y_pred_test, y_proba_test = get_predictions(final_model, X_test)
            test_metrics = compute_metrics(y_test.values, y_pred_test, y_proba_test, model_name=chosen_name)

        val_metrics = None
        if use_stored:
            val_df_raw = st.session_state.get("val_df_raw")
            if val_df_raw is not None and not val_df_raw.empty and "Activity" in val_df_raw.columns:
                with st.spinner("Generating descriptors for validation set…"):
                    X_val_raw, _ = generate_all_features(val_df_raw["SMILES"].tolist(), use_rdkit=True)
                    X_val = X_val_raw.reindex(columns=feature_cols, fill_value=0).fillna(0)
                    y_val = val_df_raw["Activity"].astype(int).reset_index(drop=True)
                    y_pred_val, y_proba_val = get_predictions(final_model, X_val)
                    val_metrics = compute_metrics(y_val.values, y_pred_val, y_proba_val,
                                                  model_name=f"{chosen_name} (Validation)")
        elif val_upload is not None:
            vdf = pd.read_csv(val_upload)
            X_vr, _ = generate_all_features(vdf["SMILES"].tolist(), use_rdkit=True)
            X_v  = X_vr.reindex(columns=feature_cols, fill_value=0).fillna(0)
            y_v  = vdf["Activity"].astype(int).reset_index(drop=True)
            yp, ypr = get_predictions(final_model, X_v)
            val_metrics = compute_metrics(y_v.values, yp, ypr, model_name=f"{chosen_name} (Validation)")

        st.session_state.update({
            "final_model": final_model, "final_model_name": chosen_name,
            "final_test_metrics": test_metrics, "final_val_metrics": val_metrics,
            "final_y_pred_test": y_pred_test, "final_y_proba_test": y_proba_test,
            "step4_complete": True,
        })
        st.success("Training and evaluation complete.")

    if not st.session_state.get("step4_complete"):
        return

    final_model   = st.session_state["final_model"]
    chosen_name   = st.session_state["final_model_name"]
    test_metrics  = st.session_state["final_test_metrics"]
    val_metrics   = st.session_state.get("final_val_metrics")
    y_pred_test   = st.session_state["final_y_pred_test"]
    y_proba_test  = st.session_state.get("final_y_proba_test")

    # Metrics table
    section_header("chart", "Performance Metrics")
    all_m = [test_metrics] + ([val_metrics] if val_metrics else [])
    mdf   = pd.DataFrame(all_m)
    float_cols = [c for c in ["Accuracy","Balanced_Accuracy","Sensitivity","Specificity","F1_Score","ROC_AUC"] if c in mdf.columns]
    display_mdf = mdf.copy()
    for c in float_cols:
        display_mdf[c] = display_mdf[c].apply(lambda x: f"{x:.4f}" if pd.notnull(x) and isinstance(x, (int, float)) else x)
        
    st.dataframe(display_mdf, use_container_width=True, hide_index=True)

    # Test set metric cards
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="metric-label">ROC-AUC</div>
        <div class="metric-value teal">{test_metrics.get('ROC_AUC','—'):.4f}</div></div>
      <div class="metric-card"><div class="metric-label">F1 Score</div>
        <div class="metric-value">{test_metrics.get('F1_Score','—'):.4f}</div></div>
      <div class="metric-card"><div class="metric-label">Sensitivity</div>
        <div class="metric-value amber">{test_metrics.get('Sensitivity','—'):.4f}</div></div>
      <div class="metric-card"><div class="metric-label">Specificity</div>
        <div class="metric-value" style="color:var(--purple);">{test_metrics.get('Specificity','—'):.4f}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # CM + ROC side by side
    section_header("chart", "Confusion Matrix & ROC Curve")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_cm_chart(y_test.values, y_pred_test, "Confusion Matrix — Test Set"), use_container_width=True)
    with col2:
        if y_proba_test is not None:
            try:
                st.plotly_chart(_roc_chart(y_test.values, y_proba_test, f"ROC Curve — {chosen_name}"), use_container_width=True)
            except Exception:
                st.info("ROC curve unavailable.")

    # SHAP
    section_header("molecule", "Feature Explainability (SHAP)")
    with st.expander("View SHAP Feature Importance"):
        _try_shap(final_model, X_train, chosen_name)

    # Export
    section_header("pkg", "Download Project Package", "Export a structured ZIP file with your model, datasets, feature list, and metrics report")
    
    if "zip_buffer" not in st.session_state:
        st.session_state["zip_buffer"] = None

    if st.session_state["zip_buffer"] is None:
        if st.button("Prepare Project Package", use_container_width=True):
            with st.spinner("Structuring files and compressing…"):
                buf = export_model_zip(
                    model=final_model, feature_columns=feature_cols,
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test,
                    metrics_dict=test_metrics,
                    model_name=chosen_name.replace(" ", "_").lower(),
                    val_df=st.session_state.get("val_df_raw"),
                )
                st.session_state["zip_buffer"] = buf
            st.rerun()
    else:
        st.download_button(
            "⬇️ Download Project Package (.zip)",
            data=st.session_state["zip_buffer"],
            file_name=f"{chosen_name.replace(' ','_').lower()}_project_package.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )
        if st.button("Repackage Data", use_container_width=True):
            st.session_state["zip_buffer"] = None
            st.rerun()
