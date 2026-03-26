# Classification Model Builder

A production-ready, interactive web application for end-to-end binary classification of molecular data (SMILES format), built with **Streamlit** and **Python**.

---

## Features

- **Step-by-Step Pipeline** with manual user control at each stage
- **SMILES Validation** using RDKit
- **Molecular Descriptors**: RDKit physicochemical, Morgan fingerprints, MACCS keys
- **Feature Selection**: Variance threshold, correlation filtering, model-based (RF importance)
- **Model Benchmarking**: Random Forest, LightGBM, XGBoost, SVM, Logistic Regression
- **Hyperparameter Tuning**: RandomizedSearchCV (5-fold CV)
- **Class Imbalance Handling**: SMOTE
- **Explainability**: SHAP summary plots for tree models
- **Export**: ZIP package with model, pipeline, datasets, and metrics report
- **Prediction Module**: Upload trained model + new SMILES → downloadable Excel predictions

---

## Setup & Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note**: RDKit may require a specific install on some systems:
> ```bash
> pip install rdkit-pypi
> ```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage Guide

| Step | Description |
|------|-------------|
| **Step 0** | Upload CSV with `SMILES` and `Activity` columns |
| **Step 1** | Data cleaning, train/test/validation split, download splits |
| **Step 2** | Generate descriptors, run feature selection, download processed features |
| **Step 3** | Benchmark multiple ML models, select best model |
| **Step 4** | Train final model, evaluate, export ZIP package |
| **Predictions** | Upload saved model + new SMILES → Excel predictions |

---

## Sample Data

A sample dataset is provided in `sample_data/sample_cardiotox.csv`.

**Format**:
```
SMILES,Activity
CC(=O)Nc1ccc(O)cc1,1
c1ccc(cc1)C(=O)O,0
...
```

---

## Project Structure

```
Classification Model Builder/
├── app.py                        # Main Streamlit app
├── requirements.txt
├── README.md
├── sample_data/
│   └── sample_cardiotox.csv
├── utils/
│   ├── smiles_validator.py
│   ├── descriptor_generator.py
│   ├── feature_selector.py
│   ├── model_trainer.py
│   ├── metrics.py
│   └── exporter.py
└── pages/
    ├── step0_upload.py
    ├── step1_preprocessing.py
    ├── step2_features.py
    ├── step3_benchmarking.py
    ├── step4_final_model.py
    └── prediction.py
```

---

## Tech Stack

- **Frontend**: Streamlit
- **Cheminformatics**: RDKit
- **ML**: scikit-learn, LightGBM, XGBoost
- **Class Balancing**: imbalanced-learn (SMOTE)
- **Explainability**: SHAP
- **Visualization**: Plotly
- **Data**: Pandas, NumPy

---

## Reproducibility

All random operations use `RANDOM_SEED = 42`.

---

## License

Internal use. Contact the developer for licensing details.
