"""
Prediction Module (Premium UI Edition)
"""

import io
import json
import streamlit as st
import pandas as pd
import joblib

from utils.descriptor_generator import generate_all_features
from utils.exporter import export_predictions_xlsx
from utils.smiles_validator import validate_smiles


def render():
    from utils.ui_components import section_header, icon

    st.markdown("""<p>Upload a <strong>trained model</strong> and <strong>feature columns JSON</strong>
    exported from Step 4, or use the model currently in session — then run predictions on any
    new SMILES dataset and download results as Excel.</p>""", unsafe_allow_html=True)

    # Model source
    section_header("pkg", "Model Source", "Load a saved model or use the one currently in session")
    use_session = st.checkbox("Use model from current session (Step 4)", value=False, key="use_session_pred")

    model, feature_cols = None, None

    if use_session:
        model        = st.session_state.get("final_model")
        feature_cols = st.session_state.get("feature_columns")
        name_s       = st.session_state.get("final_model_name", "Session Model")
        if model is None or feature_cols is None:
            st.error("No trained model found in session. Complete Step 4 first.")
        else:
            st.markdown(f"""
            <div style="display:inline-flex;align-items:center;gap:10px;background:var(--teal-dim);
                 border:1px solid rgba(6,214,160,0.3);border-radius:99px;padding:6px 16px;">
              {icon("check", 16, "#06d6a0")}
              <span style="font-size:0.84rem;font-weight:600;color:var(--teal);">Session model loaded: {name_s}</span>
            </div>""", unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            model_file = st.file_uploader("Upload model (.pkl)", type=["pkl"], key="pred_model_file")
        with c2:
            feat_file  = st.file_uploader("Upload feature_columns.json", type=["json"], key="pred_feat_file")

        if model_file:
            try:
                model = joblib.load(io.BytesIO(model_file.read()))
                st.success(f"Model loaded from file.")
            except Exception as e:
                st.error(f"Failed: {e}")

        if feat_file:
            try:
                feature_cols = json.load(feat_file)
                st.success(f"Feature columns loaded ({len(feature_cols):,} features).")
            except Exception as e:
                st.error(f"Failed: {e}")

    # SMILES input
    section_header("molecule", "New SMILES Input", "Upload a CSV with SMILES column for batch predictions")
    new_file = st.file_uploader("Upload SMILES CSV", type=["csv"], key="pred_smiles_file",
                                label_visibility="collapsed")
    if new_file is None:
        st.info("Upload a CSV with a SMILES column to continue.")
        return

    try:
        pred_df = pd.read_csv(new_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    cols      = pred_df.columns.tolist()
    smiles_col = st.selectbox("SMILES column", cols,
                              index=cols.index("SMILES") if "SMILES" in cols else 0)

    # Descriptor config
    section_header("settings", "Descriptor Settings", "Must match the settings used during training")
    with st.expander("Configure descriptors", expanded=False):
        p_rdkit  = st.checkbox("RDKit Descriptors", value=True)
        p_morgan = st.checkbox("Morgan Fingerprints", value=False)
        p_mr     = st.selectbox("Morgan Radius", [2, 3], index=0, disabled=not p_morgan)
        p_mb     = st.selectbox("Bit Length", [512, 1024, 2048], index=2, disabled=not p_morgan)
        p_maccs  = st.checkbox("MACCS Keys", value=False)

    if st.button("Run Predictions", type="primary", use_container_width=True):
        if model is None:
            st.error("No model loaded.")
            return
        if feature_cols is None:
            st.error("No feature columns provided.")
            return

        smiles_list  = pred_df[smiles_col].astype(str).tolist()
        valid_mask   = [validate_smiles(s) for s in smiles_list]
        n_invalid    = sum(not v for v in valid_mask)
        if n_invalid:
            st.warning(f"{n_invalid} invalid SMILES detected — those rows will be marked.")

        pb = st.progress(0, text="Generating descriptors…")
        X_raw, _ = generate_all_features(smiles_list,
            use_rdkit=p_rdkit, use_morgan=p_morgan, use_maccs=p_maccs,
            morgan_radius=p_mr, morgan_bits=p_mb)
        pb.progress(60, text="Aligning features…")

        X_aligned = X_raw.reindex(columns=feature_cols, fill_value=0).fillna(0)
        pb.progress(80, text="Running model…")

        y_pred  = model.predict(X_aligned)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try: y_proba = model.predict_proba(X_aligned)
            except Exception: pass

        pb.progress(100, text="Done.")

        result_df = pred_df.copy()
        result_df["Prediction"]      = y_pred.astype(int)
        result_df["Predicted_Label"] = result_df["Prediction"].map({0: "Inactive", 1: "Active"})
        if y_proba is not None:
            result_df["P(Inactive)"] = [round(p, 4) for p in y_proba[:, 0]]
            result_df["P(Active)"]   = [round(p, 4) for p in y_proba[:, 1]]

        for j, (valid, ix) in enumerate(zip(valid_mask, result_df.index)):
            if not valid:
                result_df.at[ix, "Prediction"]      = -1
                result_df.at[ix, "Predicted_Label"] = "INVALID SMILES"

        st.session_state["prediction_results"] = result_df
        st.success(f"Predictions complete for {len(result_df):,} molecules.")

    if "prediction_results" not in st.session_state:
        return

    res = st.session_state["prediction_results"]
    valid_p = res[res["Prediction"] != -1]

    # Summary cards
    section_header("chart", "Prediction Summary")
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="metric-label">Total Molecules</div>
        <div class="metric-value">{len(res):,}</div></div>
      <div class="metric-card"><div class="metric-label">Predicted Active</div>
        <div class="metric-value teal">{int((valid_p['Prediction']==1).sum()):,}</div></div>
      <div class="metric-card"><div class="metric-label">Predicted Inactive</div>
        <div class="metric-value" style="color:var(--purple);">{int((valid_p['Prediction']==0).sum()):,}</div></div>
      <div class="metric-card"><div class="metric-label">Invalid SMILES</div>
        <div class="metric-value {'red' if len(res)-len(valid_p) else 'teal'}">{len(res)-len(valid_p):,}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Distribution donut (if proba available)
    if "P(Active)" in res.columns:
        import plotly.graph_objects as go
        counts = valid_p["Prediction"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Active", "Inactive"],
            values=[counts.get(1, 0), counts.get(0, 0)],
            marker=dict(colors=["#06d6a0", "#7c6fe0"], line=dict(color="#101622", width=3)),
            hole=0.6, textfont=dict(size=13, color="#e2e8f0"),
        ))
        fig.update_layout(
            paper_bgcolor="#101622", plot_bgcolor="#101622",
            font=dict(family="Inter, sans-serif", color="#e2e8f0"),
            margin=dict(l=0, r=0, t=30, b=0), height=250,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            annotations=[dict(text=f"<b>{len(valid_p)}</b><br><span style='font-size:0.7rem'>predicted</span>",
                              font_size=18, font_color="#e2e8f0", showarrow=False)],
        )
        col_d, col_t = st.columns([1, 2])
        with col_d: st.plotly_chart(fig, use_container_width=True)

    # Table
    section_header("molecule", "Results Table")
    st.dataframe(res, use_container_width=True, hide_index=True)

    # Download
    section_header("download", "Download Predictions")
    sm_out  = res.get("SMILES", res.iloc[:, 0]).tolist() if "SMILES" in res.columns else res.iloc[:, 0].tolist()
    pred_out = res["Prediction"].tolist()
    p0 = res["P(Inactive)"].tolist() if "P(Inactive)" in res.columns else None
    p1 = res["P(Active)"].tolist()   if "P(Active)"   in res.columns else None

    buf = export_predictions_xlsx(sm_out, pred_out, p0, p1)
    st.download_button(
        "Download Predictions (.xlsx)",
        data=buf,
        file_name="molclass_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
