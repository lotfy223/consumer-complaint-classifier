# 📨 Consumer Complaint Classifier

A machine learning project that classifies consumer complaints into
five financial product categories. It includes a Streamlit demo and
experiments comparing a traditional machine learning model with GRU
and DistilBERT.

## 🎯 Project Goal

Help categorize English-language consumer complaints based on their
written narratives.

The application predicts one of five categories:

1. Credit Card
2. Credit Reporting
3. Debt Collection
4. Mortgages and Loans
5. Retail Banking

## 🌐 Live Demo

A Streamlit application allows users to enter a complaint and see the
predicted category and the model's scores for all five categories.

## 📊 Dataset

The project uses the
[US Consumer Finance Complaints dataset on Kaggle](https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints).

After filtering to the selected products and removing exact duplicate
complaints, the dataset contained **63,984 complaints**.

| Split | Number of complaints |
|---|---:|
| Training | 44,788 |
| Validation | 9,598 |
| Test | 9,598 |

Duplicate texts with conflicting category labels were excluded before
splitting. Exact deduplication does not guarantee that all similar
complaints were removed.

For this project, mortgage, consumer loan, student loan and payday loan
complaints were combined into the **Mortgages and Loans** category.

## 🤖 Models and Experiments

### Full-data baseline

The deployed model combines **TF-IDF** text features with
**Logistic Regression**. It was trained on the full training split.

| Metric | Validation | Test |
|---|---:|---:|
| Macro F1 | 0.8621 | **0.8635** |
| Accuracy | 0.8748 | 0.8746 |

Macro F1 was selected to reflect performance across all five categories,
including the smaller classes.

### GRU experiment

A GRU model trained on the full training split achieved a **validation
macro F1 of 0.8399**. It was not evaluated on the test set.

### Small-sample DistilBERT experiment

Because a GPU was unavailable, DistilBERT was trained for one epoch on
2,000 complaints and evaluated on 1,000 validation complaints. A baseline
was trained and evaluated on the same samples:

| Model | Validation macro F1 | Training time |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.7347 | 8.1 seconds |
| DistilBERT | 0.7374 | 1,580.9 seconds |

These small-sample scores should **not** be compared directly with the
full-data results. They represent one limited CPU experiment, not a
final comparison of the models.

## 🛠️ Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- PyTorch and Hugging Face Transformers
- Streamlit
- Google Colab

## ▶️ Run the App

Clone the repository, install the dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
