# Medical Claim Denial Prediction

An **ML-powered medical claim denial prediction system** using CatBoost to predict medical claim acceptance or denial and classify denied claims into 11 categories. The project aims to identify potential billing issues, understand denial patterns, and support data-driven claim review and corrective decisions.

---

## System Architecture

The project follows a two-stage machine learning approach to predict medical claim outcomes and identify reasons for claim denials.

![Medical Billing Claim Denial Prediction Flowchart](docs/Medical%20Billing%20Claim%20Denial%20Prediction%20Flowchart.png)

- **Parent Model:** Predicts whether a medical claim is accepted or denied.
- **Child Models:** Classifies denied claims into one of 11 denial categories.

The architecture illustrates the overall workflow. Payer submission and corrective-action processes are conceptual and are not implemented as automated integrations in these notebooks.

---

## Project Overview

Medical claim denials can create financial losses, administrative workload, and delays in reimbursement. This project applies machine learning to historical medical billing data to predict claim outcomes and identify possible reasons for denial.

The project consists of two complementary machine learning workflows.

### 1. Parent Model – Binary Classification

**Notebook:** `DENIAL Parent Model.ipynb`

The Parent Model predicts the outcome of a medical claim.

| Class | Meaning |
|---|---|
| 0 | Denied |
| 1 | Accepted |

**Key implementation steps:**

- Combines historical accepted and denied claim records.
- Performs data cleaning and preprocessing.
- Engineers features from CPT codes, modifiers, diagnosis codes, payer information, and other claim attributes.
- Generates statistical encoding features, including:
  - CPT rate
  - CPT and modifier rate
  - CPT and ICD rate
  - CPT pair rate
- Uses a claim-level train/test split to reduce data leakage between training and testing.

**Machine Learning Model:** CatBoost Classifier

**Training Configuration:**

| Parameter | Value |
|---|---|
| Iterations | 5,000 |
| Depth | 8 |
| Learning Rate | 0.05 |
| Loss Function | Logloss |
| Evaluation Metric | AUC |
| Early Stopping | 50 rounds |
| Random Seed | 42 |

**Reported Performance:**

| Metric | Result |
|---|---:|
| Test Accuracy | 96% |
| Best Validation AUC | 0.9962 |
| Best Iteration | 315 |

The model achieved approximately 96% test accuracy on the evaluated test split.

### 2. Child Models – Multiclass Classification

**Notebook:** `Child Models.ipynb`

The Child Model workflow focuses on identifying the reasons behind denied claims.

It processes denied claim records, standardizes rejection descriptions, and maps them into 11 broader denial categories.

**Key implementation steps:**

- Cleans and standardizes rejection descriptions.
- Groups raw rejection descriptions into broader denial categories.
- Applies a priority system when multiple denial reasons exist for the same claim and CPT.
- Engineers claim-level and CPT-related features.
- Applies probability-based feature encoding.
- Trains a CatBoost multiclass classifier.
- Evaluates model performance on later time periods.

**The 11 Denial Categories:**

1. Invalid Modifier
2. Inclusive Procedure
3. Lack of Information
4. Non-Covered Benefits
5. CPT–ICD Mismatch
6. Invalid Place of Service
7. Gender Inconsistency
8. Age Inconsistency
9. Authorization
10. Eligibility
11. Payer Rule

**Reported Forward-Testing Performance:**

| Evaluation Period | Accuracy |
|---|---:|
| December 2025 | 63.73% |
| January 2026 | 59.39% |

These results represent evaluations on later periods rather than the Parent Model's random test split.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| CatBoost | Binary and multiclass classification |
| Pandas | Data manipulation and processing |
| NumPy | Numerical computations |
| Scikit-learn | Data preprocessing and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Development and experimentation |

---

## Repository Structure

```text
Medical-Claim-Denial-Prediction/
│
├── DENIAL Parent Model.ipynb
├── Child Models.ipynb
│
├── docs/
│   └── Medical Billing Claim Denial Prediction Flowchart.png
│
├── models/
│   ├── catboost_denial_model.cbm
│   └── catboost_denial_multiclass.cbm
│
├── README.md
├── requirements.txt
└── .gitignore
```

*This is a suggested organization. Actual files and folders may differ depending on which artifacts are included in the repository.*

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/deagle-007/Medical-Claim-Denial-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Medical-Claim-Denial-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install pandas numpy scikit-learn catboost matplotlib seaborn jupyter
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open either notebook to explore the respective machine learning workflow.

---

## Model Artifacts

The notebooks generate trained models and supporting feature-encoding artifacts.

**Parent Model:**

- `catboost_denial_model.cbm`
- `lookup_tables.pkl`
- `feature_columns.pkl`
- `cat_cols.pkl`

**Child Model:**

- `catboost_denial_multiclass.cbm`
- `probability_lookup_tables.pkl`

These artifacts support model persistence and reuse. The corresponding preprocessing and feature-generation steps must also be applied when using the saved models.

---

## Dataset and Data Privacy

The models were developed using historical medical claim records, including accepted and denied claims.

The original datasets are not included in this public repository because medical billing records may contain confidential or personally identifiable information.

To reproduce the workflows, users must provide appropriately authorized and anonymized datasets with the required columns and compatible preprocessing.

---

## Limitations

- Model performance depends on the quality and distribution of historical claim data.
- The Parent Model and Child Model are developed as complementary workflows; their automatic end-to-end integration is not implemented.
- Multiclass denial classification is more challenging than binary claim outcome prediction.
- Forward-testing results may vary across payers, claim types, and evaluation periods.
- The project does not directly submit claims to insurance companies or integrate with payer systems.

---

## Future Improvements

- Integrate both models into a unified prediction pipeline.
- Develop an interactive dashboard for claim prediction and denial analysis.
- Improve multiclass classification through additional feature engineering and model optimization.
- Introduce explainability techniques to understand important denial-related features.
- Explore deployment through an API for integration with medical billing applications.

---

## License

This project is available for educational and research purposes. Add an appropriate open-source license before permitting reuse or redistribution.
