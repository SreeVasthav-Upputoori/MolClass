"""
Step 2 — Feature Engineering (Premium UI Edition)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE

from utils.descriptor_generator import generate_all_features
from utils.feature_selector import apply_feature_selection

RANDOM_SEED = 42
DARK_CHART = dict(
    paper_bgcolor="#101622", plot_bgcolor="#101622",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=60, r=30, t=50, b=40),
)


def render():
    from utils.ui_components import section_header

    if not st.session_state.get("step1_complete"):
        st.warning("Complete **Step 1 — Preprocessing** first.")
        return

    train_df = st.session_state["train_df"]
    test_df  = st.session_state["test_df"]
    val_df   = st.session_state["val_df"]
    apply_smote = st.session_state.get("apply_smote", False)

    st.markdown(f"""<p>Generating features for <strong style="color:var(--teal);">{len(train_df):,} train</strong>,
    <strong style="color:var(--amber);">{len(test_df):,} test</strong>, and
    <strong style="color:var(--purple);">{len(val_df):,} validation</strong> molecules.
    The validation set is <strong>never used</strong> during feature selection.</p>""",
    unsafe_allow_html=True)

    # Descriptor types
    section_header("flask", "Descriptor Types", "Choose which molecular representations to compute")
    with st.expander("Select descriptors", expanded=True):
        use_rdkit  = st.checkbox("RDKit Physicochemical Descriptors  (~200 features)", value=True)
        use_morgan = st.checkbox("Morgan ECFP Fingerprints", value=False)
        if use_morgan:
            mc1, mc2 = st.columns(2)
            morgan_radius = mc1.selectbox("Radius", [2, 3], index=0, help="2 = ECFP4, 3 = ECFP6")
            morgan_bits   = mc2.selectbox("Bit Length", [512, 1024, 2048], index=2)
        else:
            morgan_radius, morgan_bits = 2, 2048
        use_maccs  = st.checkbox("MACCS Keys (167 structural bits)", value=False)

    if not (use_rdkit or use_morgan or use_maccs):
        st.error("Select at least one descriptor type.")
        return

    # Feature selection
    section_header("target", "Feature Selection", "Fitted on train set only — test is aligned, validation is untouched")
    with st.expander("Selection strategy", expanded=True):
        c1, c2 = st.columns(2)
        use_var    = c1.checkbox("Variance Threshold", value=True)
        var_thresh = c1.slider("Threshold", 0.0, 0.1, 0.01, 0.005, disabled=not use_var)
        use_corr   = c2.checkbox("Pearson Correlation Filter", value=True)
        corr_thresh= c2.slider("Max |r| allowed", 0.80, 0.99, 0.95, 0.01, disabled=not use_corr)
        use_rf_sel = st.checkbox("Random Forest Importance — keep top-N features", value=False)
        n_rf_feats = st.slider("N features to retain", 20, 500, 100, 10, disabled=not use_rf_sel)

    if st.button("Generate Descriptors & Select Features", type="primary", use_container_width=True):
        pb = st.progress(0, text="Computing descriptors for train set…")

        X_train_raw, _ = generate_all_features(
            train_df["SMILES"].tolist(),
            use_rdkit=use_rdkit, use_morgan=use_morgan, use_maccs=use_maccs,
            morgan_radius=morgan_radius, morgan_bits=morgan_bits,
        )
        y_train = train_df["Activity"].astype(int).reset_index(drop=True)
        pb.progress(33, text="Computing descriptors for test set…")

        X_test_raw, _ = generate_all_features(
            test_df["SMILES"].tolist(),
            use_rdkit=use_rdkit, use_morgan=use_morgan, use_maccs=use_maccs,
            morgan_radius=morgan_radius, morgan_bits=morgan_bits,
        )
        y_test = test_df["Activity"].astype(int).reset_index(drop=True)
        pb.progress(66, text="Applying feature selection pipeline…")

        X_train_sel, X_test_sel, fs_report = apply_feature_selection(
            X_train_raw, X_test_raw, y_train,
            use_variance=use_var, variance_threshold=var_thresh,
            use_correlation=use_corr, correlation_threshold=corr_thresh,
            use_model_based=use_rf_sel, n_model_features=n_rf_feats,
        )
        pb.progress(90, text="Finalising…")

        if apply_smote:
            try:
                sm = SMOTE(random_state=RANDOM_SEED)
                Xs, ys = sm.fit_resample(X_train_sel, y_train)
                X_train_sel = pd.DataFrame(Xs, columns=X_train_sel.columns)
                y_train = pd.Series(ys, name="Activity")
                st.info(f"SMOTE applied: {len(y_train):,} train samples "
                        f"({(y_train==0).sum()} inactive / {(y_train==1).sum()} active).")
            except Exception as e:
                st.warning(f"SMOTE failed: {e}")

        pb.progress(100, text="Done.")

        st.session_state.update({
            "X_train": X_train_sel, "X_test": X_test_sel,
            "y_train": y_train, "y_test": y_test,
            "feature_columns": X_train_sel.columns.tolist(),
            "fs_report": fs_report, "val_df_raw": val_df.copy(),
            "step2_complete": True,
        })
        st.success("Feature engineering complete. Switch to the **Benchmark** tab.")

    if not st.session_state.get("step2_complete"):
        return

    fs_report = st.session_state["fs_report"]
    X_train   = st.session_state["X_train"]
    X_test    = st.session_state["X_test"]
    y_train   = st.session_state["y_train"]

    # Summary metrics
    section_header("chart", "Feature Selection Results")
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">Initial Features</div>
        <div class="metric-value">{fs_report.get('initial_features', '—'):,}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Final Features</div>
        <div class="metric-value teal">{fs_report.get('final_features', '—'):,}</div>
        <div class="metric-sub">after all filters</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Train Shape</div>
        <div class="metric-value amber">{X_train.shape[0]:,}</div>
        <div class="metric-sub">{X_train.shape[1]} features</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Test Shape</div>
        <div class="metric-value" style="color:var(--purple);">{X_test.shape[0]:,}</div>
        <div class="metric-sub">{X_test.shape[1]} features</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Steps breakdown
    if fs_report.get("steps"):
        steps_df = pd.DataFrame(fs_report["steps"])
        st.markdown("""<div style="font-size:0.78rem;font-weight:700;color:var(--text-muted);
        text-transform:uppercase;letter-spacing:0.07em;margin:1rem 0 0.5rem 0;">
        Selection Pipeline Steps</div>""", unsafe_allow_html=True)
        st.dataframe(steps_df, use_container_width=True, hide_index=True)

    # Funnel chart of feature reduction
    if fs_report.get("steps"):
        stages = [fs_report.get("initial_features", 0)] + [s.get("remaining", 0) for s in fs_report["steps"]]
        labels = ["Initial"] + [s["name"] for s in fs_report["steps"]]
        fig = go.Figure(go.Funnel(
            y=labels, x=stages,
            textinfo="value+percent initial",
            marker=dict(color=["#06d6a0","#7c6fe0","#ffb703","#f97316","#06d6a0"][:len(stages)],
                        line=dict(color="#101622", width=2)),
            connector=dict(line=dict(color="rgba(255,255,255,0.08)")),
        ))
        fig.update_layout(**DARK_CHART, height=220, title="Feature Reduction Funnel")
        st.plotly_chart(fig, use_container_width=True)

    # Downloads
    section_header("download", "Download Processed Files")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("X_train_processed.csv", X_train.to_csv(index=False).encode(),
            "X_train_processed.csv", "text/csv", use_container_width=True)
    with d2:
        st.download_button("X_test_processed.csv", X_test.to_csv(index=False).encode(),
            "X_test_processed.csv", "text/csv", use_container_width=True)
    with d3:
        st.download_button("validation_raw.csv",
            st.session_state["val_df_raw"].to_csv(index=False).encode(),
            "validation_raw.csv", "text/csv", use_container_width=True)

    st.success("Step 2 complete. Switch to the **Benchmark** tab.")
