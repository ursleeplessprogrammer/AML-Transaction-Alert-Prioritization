# 🏦 AML Alert Triage with SAML-D

<p align="center">
      <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
      <img src="https://img.shields.io/badge/IDE-Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white" />
</p>

> **Explore synthetic banking transactions, compare machine-learning approaches, and prioritize records for human review.**

## Description

This project explores anti-money laundering (AML) transaction monitoring using the **SAML-D synthetic transaction dataset**. It focuses on examining transaction patterns, preparing data, training classification models, evaluating results under severe class imbalance, and producing an explainable review queue.

The goal is to investigate whether machine-learning methods can help prioritize transactions for analyst attention—not to determine that a person has committed a crime or to replace a bank’s compliance process.

SAML-D contains approximately **9.5 million synthetic transactions**, 12 transaction features, and 28 typologies: 11 normal and 17 suspicious. The dataset description reports that suspicious transactions represent about **0.1039%** of the data. It also describes 15 graphical transaction-network structures. Check the dataset version and documentation you download, as the Kaggle dataset may be updated. ([kaggle.com](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml))

## 🎯 Project Goals

- Explore transaction behavior and suspicious-activity labels.
- Understand how class imbalance affects AML classification.
- Build baseline models and compare their results.
- Evaluate using metrics that reveal missed cases and false alerts.
- Rank transactions for review and show the signals behind each alert.
- Document the limits of synthetic data and model-generated scores.

## 📊 Dataset

SAML-D was introduced as a synthetic transaction-monitoring dataset to support AML research, where access to real transaction data is restricted. Its typologies were informed by existing datasets, literature, and AML specialist input. It is designed for research and model comparison—not as a substitute for real-world validation. ([eprints.bournemouth.ac.uk](https://eprints.bournemouth.ac.uk/40982/1/Full_IEEE_Dataset_Conference_Paper%20%284%29.pdf))

### Dataset fields

The Kaggle dataset describes fields including:

- `Time` and `Date` — transaction time information
- `Sender_account` and `Receiver_account` — synthetic account identifiers
- `Amount` — transaction value
- `Payment_currency` and `Received_currency` — currencies involved
- `Sender_bank_location` and `Receiver_bank_location` — bank locations
- `Payment_type` — payment method
- `Is_laundering` — synthetic binary suspicious-activity label
- `Laundering_type` — transaction typology label

Confirm the actual CSV columns and label values after downloading the dataset. Do not assume the label meaning from the column name alone.

### Important dataset considerations

- The data is **synthetic**. A model may learn the simulator’s patterns rather than patterns found in real financial institutions.
- The positive class is extremely rare. Accuracy alone may look high even if the model misses most suspicious examples.
- Transaction and typology labels describe simulated cases; they are not findings about real people.
- The full dataset is large. Use an appropriately sized sample for initial development, then document how it was selected.
- Account identifiers are useful for linking transaction networks, but usually should not be treated as ordinary numeric model features.

## 🧭 Project Workflow

### 1. Inspect the data

Check:

- CSV shape, columns, types, and missing values
- Label counts and suspicious-class proportion
- Duplicate records and malformed values
- Amount and timestamp distributions
- Typology and payment-type distributions
- Sender–receiver relationships and network structure

### 2. Define the prediction task

Decide whether the model predicts:

- `Is_laundering` — binary suspicious versus normal classification, or
- `Laundering_type` — classification among transaction typologies

These are different tasks. For binary detection, do not include `Laundering_type` as an input feature: it can reveal the outcome and cause target leakage.

### 3. Create features

Possible transaction-level features include:

- Amount and a log-transformed amount
- Hour, day of week, or time of day
- Payment type
- Currency and location combinations
- Whether sender and receiver locations differ
- Sender and receiver activity counts over a defined time window
- Time since the account’s previous transaction
- Number and value of transfers involving an account
- Network features such as degree or connected components

Use only information that would have been available when the transaction was being assessed. For example, calculating an account’s total activity using transactions from the entire dataset may leak future information into a prediction.

### 4. Establish a baseline

Start with a simple, clearly documented model, such as:

- Logistic regression
- A decision tree
- Random forest
- Gradient-boosted trees

Compare models using the same data split, feature definitions, and metrics.

### 5. Evaluate and review

Use a test strategy suited to the data. If transactions have a meaningful chronological order, a time-based split is generally more realistic than randomly mixing past and future records. If multiple records belong to the same account or simulated scheme, consider whether related transactions should be grouped during validation.

Review false positives and false negatives where possible. A model score is a prioritization signal, not a conclusion.

## 🧰 Suggested Technology Stack

| Area | Tools |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Exploration and visualization | Matplotlib, Seaborn |
| Machine learning | scikit-learn |
| Graph exploration | NetworkX |
| Optional API | FastAPI |
| Optional review dashboard | Streamlit |
| Experiment tracking | MLflow or documented experiment logs |

## 📁 Project Structure

```text
aml-alert-triage/
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── features.py
│   ├── model.py
│   └── main.py
├── requirements.txt
└── README.md
```

Avoid committing a large dataset unless its license and the repository’s storage rules permit it. Document where to obtain the data instead.

## 📏 Evaluation Metrics

Because the suspicious class is rare, **accuracy should not be the headline metric**.

Consider:

- **Precision:** Of transactions flagged, what fraction are labeled suspicious?
- **Recall:** Of labeled suspicious transactions, what fraction are flagged?
- **F1-score:** A combined precision–recall measure.
- **Average precision / PR-AUC:** Summarizes precision–recall ranking performance and is useful to inspect for rare positive classes.
- **Confusion matrix:** Shows true positives, false positives, true negatives, and false negatives.
- **Alert volume:** How many records would analysts need to review at a chosen threshold?
- **Precision at a review limit:** Among the top-ranked alerts, what fraction are labeled positive?

Report the threshold and the number of alerts alongside the metrics. A threshold changes the number of alerts, precision, and recall.

## 🚨 Alert Triage Output

A useful review queue can include:

| Field | Purpose |
|---|---|
| `alert_id` | Synthetic, stable identifier for the review record |
| `transaction_id` | Reference to the transaction, if available |
| `triage_score` | Model ranking score; not necessarily a calibrated probability |
| `review_status` | Workflow status, such as `needs_human_review` |
| `review_signals` | Features or rules that contributed to prioritization |
| `model_version` | Identifies the model used |
| `created_at` | Time the alert was produced |

Example:

```json
{
  "alert_id": "SAML-D-000042",
  "transaction_id": "TX-000042",
  "triage_score": 0.82,
  "review_status": "needs_human_review",
  "review_signals": [
    "unusual amount relative to the training data",
    "cross-border transaction pattern"
  ],
  "model_version": "baseline-logistic-regression"
}
```

A score such as `0.82` should not be described as an 82% chance of money laundering unless probability calibration has been tested and supports that interpretation.

## 🛡️ Responsible Use

- Treat alerts as candidates for investigation, not evidence of wrongdoing.
- Do not automatically block, report, or accuse a person based only on this prototype.
- Keep human review and documented review procedures in the loop.
- Avoid exposing account-level data in public outputs.
- Store only the data needed for the project and follow applicable dataset terms.
- Document label definitions, sampling, feature creation, thresholds, and model versions.
- Test for data leakage, including leakage through typology labels, account aggregates, and future transactions.
- Describe results as performance on a **synthetic benchmark**, not real-world banking performance.

## 📚 Citation

If you use SAML-D, cite the dataset paper:

> B. Oztas, D. Cetinkaya, F. Adedoyin, M. Budka, H. Dogan, and G. Aksu, “Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset,” *2023 IEEE International Conference on e-Business Engineering (ICEBE)*, Sydney, Australia, 2023, pp. 47–54. DOI: [10.1109/ICEBE59045.2023.00028](https://doi.org/10.1109/ICEBE59045.2023.00028).

The dataset page lists the dataset under the **CC BY-NC-SA 4.0** license. Review the license and dataset terms before using, redistributing, or including the data in another project. ([kaggle.com](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml))

## 📝 Project Limitations

- SAML-D is synthetic and cannot establish performance on real bank transactions.
- Labels describe simulated suspicious patterns, not adjudicated real-world cases.
- A random split can create an overly optimistic estimate if related activity crosses between train and test sets.
- Transaction-only features may miss network-level behavior.
- Thresholds and scores require context and human review.
- This project is an educational prototype, not an operational AML monitoring system.

## 🔗 References

- [SAML-D on Kaggle](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
- [SAML-D paper record](https://eprints.bournemouth.ac.uk/40982/)
- [SAML-D author repository](https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D)
- [scikit-learn: Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---
## 📄 License

Please refer to the repository for applicable licensing information.

## 👨‍💻 Author

**ursleeplessprogrammer**

*This project explores how machine learning can support AML alert prioritization using synthetic data. It does not determine whether real-world activity is criminal and should not be used as a standalone compliance or enforcement system.*
