"""
metrics.py
----------
Compute classification metrics and build Plotly visualisations.
Metrics: Accuracy, Balanced Accuracy, Sensitivity, Specificity, F1, ROC-AUC
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from typing import Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> dict:
    """
    Compute all classification metrics.

    Returns dict with keys:
      Model, Accuracy, Balanced_Accuracy, Sensitivity, Specificity, F1, ROC_AUC
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = round(roc_auc_score(y_true, y_proba), 4)
        except Exception:
            roc_auc = None

    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Balanced_Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "F1_Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": roc_auc,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Plotly heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = ["Inactive (0)", "Active (1)"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[
                [0, "#f5f0e8"],
                [0.5, "#8fbc8f"],
                [1, "#3d6b35"],
            ],
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20, "color": "#2c2c2c"},
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        font=dict(family="Inter, sans-serif", size=13),
        paper_bgcolor="#faf6f0",
        plot_bgcolor="#faf6f0",
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Plotly ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc:.3f})",
            line=dict(color="#5a8a5a", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="#c9b99a", dash="dash", width=1.5),
        )
    )
    fig.update_layout(
        title=f"ROC Curve — {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        font=dict(family="Inter, sans-serif", size=13),
        paper_bgcolor="#faf6f0",
        plot_bgcolor="#faf6f0",
        xaxis=dict(gridcolor="#e8e0d0"),
        yaxis=dict(gridcolor="#e8e0d0"),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing all models across all metrics.
    metrics_df: rows = models, columns include Accuracy, Balanced_Accuracy, etc.
    """
    metric_cols = ["Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "F1_Score", "ROC_AUC"]
    palette = ["#5a8a5a", "#8fbc8f", "#c9b99a", "#a0785a", "#6b4f3a", "#b5cca0"]

    fig = go.Figure()
    for i, metric in enumerate(metric_cols):
        if metric not in metrics_df.columns:
            continue
        fig.add_trace(
            go.Bar(
                name=metric.replace("_", " "),
                x=metrics_df["Model"].tolist(),
                y=metrics_df[metric].tolist(),
                marker_color=palette[i % len(palette)],
            )
        )

    fig.update_layout(
        barmode="group",
        title="Model Comparison — All Metrics",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05], gridcolor="#e8e0d0"),
        font=dict(family="Inter, sans-serif", size=13),
        paper_bgcolor="#faf6f0",
        plot_bgcolor="#faf6f0",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig
