# Customer Transaction Prediction Dashboard

This repository contains a professional interactive dashboard built using **Python Dash**, **Plotly**, and **Dash Bootstrap Components** to visualize data characteristics and compare machine learning model performances for predicting future customer transactions.

## Folder Structure

```
Customer_Transaction_Dashboard/
│
├── app.py                     # Main dashboard entrypoint and layout definitions
├── assets/
│      style.css               # Theme definitions and layouts (supports Dark Mode)
│      favicon.png             # Site favicon
│
├── data/
│      train_subset.csv        # Downsampled training data for fast visualization rendering
│      precomputed_metrics.json # Precomputed performance metrics for classifiers
│
├── models/
│      scaler.pkl              # Fitted StandardScaler instance
│      saved_model.pkl         # Trained ensemble model instance (XGBoost/GB)
│
├── utils/
│      helper.py               # Plotly figure generation and metric parsing helpers
│
└── README.md                  # Project documentation (this file)
```

## Setup & Execution Instructions

### Prerequisites
Make sure you have python installed along with the required libraries:
```bash
pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn
```

### Steps to Run:
1. Ensure you have run the precomputation steps to create model artifacts:
   ```bash
   python precompute.py
   ```
2. Navigate to the dashboard directory:
   ```bash
   cd Customer_Transaction_Dashboard
   ```
3. Launch the Dash web server:
   ```bash
   python app.py
   ```
4. Open your web browser and navigate to: [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

## Dashboard Features

- **Dashboard**: High-level summary, KPI metric cards, and project objectives.
- **Data Overview**: Dimensions summary, target distributions (pie and bar charts), and class imbalance descriptions.
- **Feature Analysis**: Statistical summary table (Mean, Median, Standard Deviation, Min, Max), interactive Boxplots and Histogram/Density plots for any chosen anonymized feature, and correlation heatmap showing feature independence.
- **Model Performance**: Joint ROC curves, performance metric comparatives (Accuracy, Precision, Recall, F1 Score, ROC-AUC), dynamic confusion matrices, and detailed classification reports for Logistic Regression, Random Forest, Calibrated Linear SVM, and XGBoost/Gradient Boosting.
- **Business Insights**: Translates machine learning outputs into concrete bank operations scenarios (Targeted marketing, cash reserve management, fraud checks).
- **Extra Features**: Dark Mode toggle, dynamic CSV download of model performance metrics, and responsive desktop-first layout styling.
