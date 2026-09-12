# Customer Transaction Prediction

This is a comprehensive Data Science capstone project focused on predicting whether a customer will make a transaction in the future. The project includes exploratory data analysis, class imbalance handling, model training/evaluation, and a professional interactive Dash analytics application.

## Dashboard Preview & Screenshots

### 1. Dashboard Home & System Overview
![Dashboard Home](./Customer%20Transaction%20Prediction/DS6-Project-1.png)

### 2. Data Overview & Target Distribution
![Data Overview](./Customer%20Transaction%20Prediction/DS6-Project-2.png)

### 3. Feature Analysis & Distribution Plots
![Feature Analysis 1](./Customer%20Transaction%20Prediction/DS6-Project-3.png)

![Feature Analysis 2](./Customer%20Transaction%20Prediction/DS6-Project-4.png)

### 4. Correlation Analysis
![Correlation Analysis](./Customer%20Transaction%20Prediction/DS6-Project-5.png)

### 5. Model Performance & ROC Curves
![Model Performance 1](./Customer%20Transaction%20Prediction/DS6-Project-6.png)

![Model Performance 2](./Customer%20Transaction%20Prediction/DS6-Project-7.png)

### 6. Business Insights & Banking Strategy
![Business Insights](./Customer%20Transaction%20Prediction/DS6-Project-8.png)

---

## Project Structure

```
Customer-Transaction-Prediction/
│
├── Customer_Transaction_Prediction.ipynb  # Step-by-step model pipeline notebook
├── precompute.py                           # Helper script to train models and extract statistics
├── .gitignore                              # Prevents heavy dataset & model file uploads
│
├── Customer Transaction Prediction/        # Dashboard UI Screenshots
│      ├── DS6-Project-1.png
│      ├── DS6-Project-2.png
│      ├── DS6-Project-3.png
│      ├── DS6-Project-4.png
│      ├── DS6-Project-5.png
│      ├── DS6-Project-6.png
│      ├── DS6-Project-7.png
│      └── DS6-Project-8.png
│
├── Customer_Transaction_Dashboard/         # Interactive Python Dash web app
│      ├── app.py                           # Main layout routing and app callbacks
│      ├── assets/
│      │      ├── style.css                 # Clean custom theme styling (supports Dark Mode)
│      │      └── favicon.png               # Web app page icon
│      ├── data/
│      │      ├── precomputed_metrics.json  # Pre-extracted metrics for rapid UI rendering
│      │      ├── train_subset.csv          # 10,000-row subset for fast Plotly charts
│      │      ├── train.csv                 # Local lightweight sample train dataset
│      │      └── test.csv                  # Local lightweight sample test dataset
│      ├── models/
│      │      ├── scaler.pkl                # Standardized scaling object
│      │      └── saved_model.pkl           # Saved Gradient Boosting ensemble model
│      └── utils/
│             └── helper.py                 # Dynamic Plotly visualizations generator
│
└── README.md                               # Root project documentation (this file)
```

## Machine Learning Pipeline Summary

1. **Exploratory Data Analysis**: Inspected target class ratios (~10% transactions, indicating heavy imbalance), feature distributions (normally distributed), and correlations (features are highly independent).
2. **Preprocessing**: Removed ID columns, split dataset into stratified train/test segments (80/20 ratio), and normalized feature scales using `StandardScaler`.
3. **Imbalance Handling**: Addressed the 90-10 target ratio by integrating class weighting (`class_weight='balanced'` and `scale_pos_weight=9`) directly into the loss function, avoiding high-memory synthetic generation methods (SMOTE).
4. **Model Comparison**: Trained Logistic Regression, Random Forest, XGBoost (Gradient Boosting), and Calibrated Linear SVM.
5. **Evaluation**: Scored models across Accuracy, Precision, Recall, F1 Score, and ROC-AUC, selecting the Boosting classifier as the production candidate.

## Installation & Setup

Ensure Python is installed along with the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn dash dash-bootstrap-components plotly
```

### 1. Run the Jupyter Notebook
Open and run all cells in `Customer_Transaction_Prediction.ipynb` to inspect the analysis, data modeling steps, and training outputs:
```bash
jupyter notebook Customer_Transaction_Prediction.ipynb
```

### 2. Launch the Interactive Dashboard
The Dash dashboard allows you to explore metrics, check individual feature stats, toggle dark mode, and view correlation heatmaps:
```bash
python Customer_Transaction_Dashboard/app.py
```
Open your browser and navigate to: **http://127.0.0.1:8050/**

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost / Gradient Boosting** | **0.842** | **0.370** | **0.675** | **0.478** | **0.852** |
| **Logistic Regression** | 0.776 | 0.283 | 0.781 | 0.415 | 0.849 |
| **Linear SVM** | 0.774 | 0.281 | 0.783 | 0.414 | 0.849 |
| **Random Forest** | 0.825 | 0.329 | 0.601 | 0.425 | 0.814 |

*Note: The values above represent metrics obtained on our validation sets.*

## Developer
- **GitHub Repository**: [DS-Project-06-Customer-Transaction-Prediction](https://github.com/Hetk8406/DS-Project-06-Customer-Transaction-Prediction)
- **GitHub Profile**: [Hetk8406](https://github.com/Hetk8406)
- **LinkedIn**: [Het Patel](https://linkedin.com/)