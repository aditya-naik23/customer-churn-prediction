\# Customer Churn Predictor



A machine learning model that predicts whether a telecom customer is likely to churn,

based on account and service details. Built and compared six classifiers on the

Telco Customer Churn dataset; deployed the best-performing one as an interactive web app.



\## Live demo

\[Add your Streamlit app link here once deployed]



\## Models compared



| Model | Accuracy |

|---|---|

| Logistic Regression | 0.813 |

| SVM | 0.804 |

| Random Forest | 0.797 |

| KNN | 0.768 |

| Naive Bayes | 0.758 |

| Decision Tree | 0.728 |



\*\*Chosen model: Logistic Regression\*\* — highest accuracy, and interpretable

(feature coefficients show which factors push predictions toward churn),

which matters for a business use case like this.



\*Note: accuracy alone doesn't tell the full story on an imbalanced target like churn —

\[add your recall/F1 findings for the churn=1 class here once you've checked them].\*



\## How to run locally



\\`\\`\\`bash

pip install -r requirements.txt

streamlit run app.py

\\`\\`\\`



\## Files



\- `coustmers.ipynb` — full notebook: EDA, preprocessing, model training/comparison

\- `app.py` — Streamlit web app for live predictions

\- `churn\_model.pkl`, `scaler.pkl`, `encoders.pkl` — saved trained artifacts

\- `requirements.txt` — dependencies



\## Dataset



\[Add the dataset source/link here]

