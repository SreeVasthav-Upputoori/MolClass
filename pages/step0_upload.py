"""
Step 0 — Upload & Validation (Premium UI Edition)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.smiles_validator import validate_dataframe
from utils.ui_components import icon, section_header

DARK_CHART = dict(
    paper_bgcolor="#101622",
    plot_bgcolor="#101622",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)


def render():

    section_header("upload", "Upload & Validate Data",
                   "Upload a CSV with SMILES and Activity columns. The pipeline will validate each molecule using RDKit.")

    # Upload zone
    col_up, col_sample = st.columns([5, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Drag & drop your CSV file here, or click Browse",
            type=["csv"],
            key="upload_csv",
            label_visibility="collapsed",
        )
    with col_sample:
        st.markdown("<div style='padding-top:0.35rem;'></div>", unsafe_allow_html=True)
        use_sample = st.button("Use Sample Data", use_container_width=True, key="use_sample_btn")

    if use_sample:
        try:
            sample_df = pd.read_csv("sample_data/sample_cardiotox.csv")
            st.session_state["raw_df"] = sample_df
            st.session_state["upload_done"] = True
            st.success("Sample dataset loaded — 102 molecules ready for validation.")
        except FileNotFoundError:
            st.error("sample_data/sample_cardiotox.csv not found.")
            return

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["raw_df"] = df
            st.session_state["upload_done"] = True
            st.success(f"File uploaded — {len(df):,} rows detected.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

    if not st.session_state.get("upload_done"):
        # Format guide
        st.markdown("""
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem 2rem;margin-top:1rem;">
          <div style="font-size:0.78rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;">Required CSV Format</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#06d6a0;line-height:1.9;">
            SMILES,Activity<br>
            CC(=O)Nc1ccc(O)cc1,1<br>
            c1ccc(cc1)C(=O)O,0<br>
            ...
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state["raw_df"]

    # Column detection
    section_header("settings", "Column Detection", "Map your CSV columns to the expected SMILES and Activity fields")
    cols = df.columns.tolist()
    c1, c2 = st.columns(2)
    with c1:
        smiles_col = st.selectbox("SMILES column", cols,
            index=cols.index("SMILES") if "SMILES" in cols else 0, key="smiles_col")
    with c2:
        activity_col = st.selectbox("Activity column", cols,
            index=cols.index("Activity") if "Activity" in cols else (1 if len(cols) > 1 else 0), key="activity_col")

    if st.button("Run Validation", type="primary", use_container_width=True):
        with st.spinner("Validating SMILES strings with RDKit…"):
            df_std = df.rename(columns={smiles_col: "SMILES", activity_col: "Activity"})
            valid_df, invalid_df, report = validate_dataframe(df_std, "SMILES", "Activity")

        st.session_state.update({
            "valid_df": valid_df, "invalid_df": invalid_df,
            "validation_report": report, "validation_done": True,
        })

    if not st.session_state.get("validation_done"):
        return

    valid_df  = st.session_state["valid_df"]
    invalid_df = st.session_state["invalid_df"]
    report    = st.session_state["validation_report"]

    # Metrics row
    section_header("chart", "Validation Summary", "Row-level quality report")
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">Total Rows</div>
        <div class="metric-value">{report['total_rows']:,}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Valid Rows</div>
        <div class="metric-value teal">{report['valid_rows']:,}</div>
        <div class="metric-sub">{report['valid_rows']/max(report['total_rows'],1):.1%} pass rate</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Invalid / Skipped</div>
        <div class="metric-value {'red' if report['invalid_rows'] else 'teal'}">{report['invalid_rows']}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Duplicate SMILES</div>
        <div class="metric-value {'amber' if report['duplicate_smiles'] else 'teal'}">{report['duplicate_smiles']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Class distribution
    class_dist = report.get("class_distribution", {})
    if class_dist:
        section_header("molecule", "Class Distribution", "Balance between active (1) and inactive (0) compounds")
        col_tbl, col_chart = st.columns([1, 2])
        with col_tbl:
            dist_df = pd.DataFrame([
                {"Activity": "Active (1)", "Count": class_dist.get(1, 0),
                 "Fraction": f"{class_dist.get(1,0)/max(report['valid_rows'],1):.1%}"},
                {"Activity": "Inactive (0)", "Count": class_dist.get(0, 0),
                 "Fraction": f"{class_dist.get(0,0)/max(report['valid_rows'],1):.1%}"},
            ])
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
            imr = report.get("imbalance_ratio")
            if imr is not None:
                if imr > 3:
                    st.warning(f"Imbalance ratio (Inactive/Active): **{imr:.2f}** — consider SMOTE in Step 1.")
                else:
                    st.success(f"Imbalance ratio: **{imr:.2f}** — acceptable balance.")

        with col_chart:
            fig = go.Figure(go.Pie(
                labels=["Active (1)", "Inactive (0)"],
                values=[class_dist.get(1, 0), class_dist.get(0, 0)],
                marker=dict(colors=["#06d6a0", "#7c6fe0"], line=dict(color="#101622", width=3)),
                hole=0.55,
                textfont=dict(size=13, color="#e2e8f0"),
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            ))
            fig.update_layout(
                **DARK_CHART,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                            font=dict(color="#e2e8f0")),
                height=280,
                annotations=[dict(text=f"<b>{report['valid_rows']}</b>", font_size=22,
                                  font_color="#e2e8f0", showarrow=False)],
            )
            st.plotly_chart(fig, use_container_width=True)

    # Invalid rows
    if not invalid_df.empty:
        section_header("alert", "Problematic Rows", "Rows excluded from the pipeline", "var(--amber-dim)", "#ffb703")
        st.dataframe(invalid_df, use_container_width=True, hide_index=True)
    else:
        st.success("All rows passed validation — no invalid SMILES detected.")

    # Preview
    section_header("upload", "Data Preview", "First 20 valid rows")
    st.dataframe(valid_df.head(20), use_container_width=True, hide_index=True)

    # Gate status
    if report["valid_rows"] < 20:
        st.error("Too few valid rows (minimum 20 required) to continue.")
    else:
        st.session_state["step0_complete"] = True
        st.success(f"{report['valid_rows']:,} valid rows ready. Switch to the **Preprocess** tab to continue.")
