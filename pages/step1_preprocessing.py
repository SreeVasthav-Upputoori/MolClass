"""
Step 1 — Data Preprocessing (Premium UI Edition)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
DARK_CHART = dict(
    paper_bgcolor="#101622", plot_bgcolor="#101622",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
)


def _class_bar(df: pd.DataFrame, title: str) -> go.Figure:
    counts = df["Activity"].astype(int).value_counts().sort_index()
    val_0 = counts.get(0, 0)
    val_1 = counts.get(1, 0)
    max_val = max(val_0, val_1)
    
    fig = go.Figure(go.Bar(
        x=["Inactive", "Active"],
        y=[val_0, val_1],
        marker=dict(
            color=["#7c6fe0", "#06d6a0"],
            line=dict(color="#101622", width=1),
        ),
        text=[val_0, val_1],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color="#e2e8f0", size=13, family="Inter"),
    ))
    fig.update_layout(
        **DARK_CHART,
        title=dict(text=title, font=dict(size=13, color="#7a8fa6"), x=0.5, xanchor="center"),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", 
            zerolinecolor="rgba(255,255,255,0.08)",
            range=[0, max_val * 1.2] if max_val > 0 else [0, 1]
        ),
        xaxis=dict(color="#7a8fa6"),
        height=240,
        showlegend=False,
    )
    return fig


def render():
    from utils.ui_components import icon, section_header

    if not st.session_state.get("step0_complete"):
        st.warning("Complete **Step 0 — Upload & Validate** first.")
        return

    valid_df = st.session_state.get("valid_df", pd.DataFrame())
    if valid_df.empty:
        st.error("No valid data found. Re-run Step 0.")
        return

    st.markdown(f'<p>Working with <strong style="color:var(--teal);">{len(valid_df):,} valid molecules</strong> from your dataset.</p>', unsafe_allow_html=True)

    # Cleaning
    section_header("settings", "Cleaning Options")
    with st.expander("Configure cleaning", expanded=True):
        c1, c2 = st.columns(2)
        remove_dupes = c1.checkbox("Remove duplicate SMILES", value=True)
        standardize  = c2.checkbox("Canonicalize SMILES (RDKit)", value=False,
                                   help="Slower, converts all SMILES to canonical form.")

    # Split config
    section_header("chart", "Split Configuration", "Partition into train, test, and held-out validation sets")
    with st.expander("Configure split ratios", expanded=True):
        c1, c2, c3 = st.columns(3)
        train_pct = c1.slider("Train %", 50, 85, 70, 5, key="train_pct")
        remaining = 100 - train_pct
        test_pct  = c2.slider("Test %", 5, remaining - 5, min(15, remaining // 2), 5, key="test_pct")
        val_pct   = remaining - test_pct
        c3.metric("Validation %", val_pct)
        stratify  = st.checkbox("Stratified split (preserves class ratio)", value=True)

    # Handling Imbalance
    section_header("molecule", "Class Imbalance", "Balance your training data before feature extraction")
    with st.expander("Imbalance handling", expanded=False):
        balance_method = st.radio("Training set balancing method:", ["None", "Undersample majority class (fast)"], 
                                  help="Undersampling balances the train set by randomly dropping majority rows, saving time on descriptor generation.")


    if st.button("Run Preprocessing", type="primary", use_container_width=True):
        df = valid_df.copy()

        if remove_dupes:
            before = len(df)
            df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
            removed = before - len(df)
            if removed:
                st.info(f"Removed {removed} duplicate SMILES.")

        if standardize:
            from rdkit import Chem
            def canon(smi):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    return Chem.MolToSmiles(mol) if mol else smi
                except Exception:
                    return smi
            with st.spinner("Canonicalizing SMILES…"):
                df["SMILES"] = df["SMILES"].apply(canon)

        y   = df["Activity"].astype(int)
        idx = df.index.tolist()
        test_val_ratio = (test_pct + val_pct) / 100.0

        try:
            strat = y if stratify else None
            tr_idx, tmp_idx = train_test_split(idx, test_size=test_val_ratio,
                                               random_state=RANDOM_SEED, stratify=strat)
            val_ratio_of_tmp = val_pct / (test_pct + val_pct)
            strat_tmp = y.loc[tmp_idx] if stratify else None
            te_idx, va_idx = train_test_split(tmp_idx, test_size=val_ratio_of_tmp,
                                              random_state=RANDOM_SEED, stratify=strat_tmp)
        except Exception as e:
            st.error(f"Split failed: {e}. Try disabling stratified split.")
            return

        train_df = df.loc[tr_idx].reset_index(drop=True)
        test_df  = df.loc[te_idx].reset_index(drop=True)
        val_df   = df.loc[va_idx].reset_index(drop=True)

        # Apply Undersampling to Train Set if requested
        if balance_method == "Undersample majority class (fast)":
            min_class_count = train_df["Activity"].value_counts().min()
            train_labels = train_df["Activity"].unique()
            balanced_dfs = []
            for lbl in train_labels:
                sub_df = train_df[train_df["Activity"] == lbl]
                if len(sub_df) > min_class_count:
                    sub_df = sub_df.sample(n=min_class_count, random_state=RANDOM_SEED)
                balanced_dfs.append(sub_df)
            train_df = pd.concat(balanced_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
            st.info(f"Undersampling applied. Train set perfectly balanced at {min_class_count} per class.")

        st.session_state.update({
            "train_df": train_df, "test_df": test_df, "val_df": val_df,
            "step1_complete": True,
        })
        st.success("Preprocessing complete. Proceed to the Features tab.")

    if not st.session_state.get("step1_complete"):
        return

    train_df = st.session_state["train_df"]
    test_df  = st.session_state["test_df"]
    val_df   = st.session_state["val_df"]
    total    = len(train_df) + len(test_df) + len(val_df)

    # Stats
    section_header("chart", "Split Results")
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">Total (clean)</div>
        <div class="metric-value">{total:,}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Train</div>
        <div class="metric-value teal">{len(train_df):,}</div>
        <div class="metric-sub">{len(train_df)/total:.0%} of dataset</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Test</div>
        <div class="metric-value amber">{len(test_df):,}</div>
        <div class="metric-sub">{len(test_df)/total:.0%} of dataset</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Validation</div>
        <div class="metric-value" style="color:var(--purple);">{len(val_df):,}</div>
        <div class="metric-sub">{len(val_df)/total:.0%} · held out</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts
    section_header("molecule", "Class Distribution per Split")
    col1, col2, col3 = st.columns(3)
    with col1: st.plotly_chart(_class_bar(train_df, "Train Set"), use_container_width=True)
    with col2: st.plotly_chart(_class_bar(test_df,  "Test Set"),  use_container_width=True)
    with col3: st.plotly_chart(_class_bar(val_df,   "Validation Set"), use_container_width=True)

    # Downloads
    section_header("download", "Download Split Files")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("Download train.csv", train_df.to_csv(index=False).encode(),
            "train.csv", "text/csv", use_container_width=True)
    with d2:
        st.download_button("Download test.csv", test_df.to_csv(index=False).encode(),
            "test.csv", "text/csv", use_container_width=True)
    with d3:
        st.download_button("Download validation.csv", val_df.to_csv(index=False).encode(),
            "validation.csv", "text/csv", use_container_width=True)

    st.success("Step 1 complete. Switch to the **Features** tab.")
