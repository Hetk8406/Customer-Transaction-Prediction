import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

print("Starting precomputation script...")

# 1. Create directory structure
dirs = [
    "Customer_Transaction_Dashboard",
    "Customer_Transaction_Dashboard/assets",
    "Customer_Transaction_Dashboard/data",
    "Customer_Transaction_Dashboard/models",
    "Customer_Transaction_Dashboard/utils"
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created directory: {d}")

# 2. Load the dataset
data_path = "Data/train(1).csv"
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)
print(f"Data loaded. Shape: {df.shape}")

# 3. Save a small subset for the dashboard visualizations to load quickly
subset_df = df.sample(n=10000, random_state=42)
subset_df.to_csv("Customer_Transaction_Dashboard/data/train_subset.csv", index=False)
print("Saved 10,000-row subset for visualization.")

# 4. Prepare data for model training
# Use a 30,000-row stratified subset to train models quickly but realistically
train_subset = df.sample(n=30000, random_state=42, weights=df['target'].map({0: 1.0, 1: 3.0})) # slightly boost positive class representation for quick training
X = train_subset.drop(columns=['ID_code', 'target'])
y = train_subset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and models
with open("Customer_Transaction_Dashboard/models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 5. Train models and get metrics
models = {}

# Model 1: Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr

# Model 2: Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=50, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
models['Random Forest'] = rf

# Model 3: XGBoost / Gradient Boosting
print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
gb.fit(X_train_scaled, y_train)
models['XGBoost/GB'] = gb

# Model 4: Linear SVM
print("Training Calibrated SVM...")
svm_base = LinearSVC(class_weight='balanced', dual=False, random_state=42, max_iter=2000)
svm = CalibratedClassifierCV(svm_base)
svm.fit(X_train_scaled, y_train)
models['Linear SVM'] = svm

# Save the best model (Gradient Boosting/XGBoost)
with open("Customer_Transaction_Dashboard/models/saved_model.pkl", "wb") as f:
    pickle.dump(gb, f)
print("Saved best model (Gradient Boosting) to saved_model.pkl")

# Generate metrics dictionary
precomputed_metrics = {}

for name, model in models.items():
    print(f"Calculating metrics for {name}...")
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    # classification report dict
    rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    # Downsample ROC points to keep file small
    step = max(1, len(fpr) // 100)
    fpr_ds = fpr[::step].tolist()
    tpr_ds = tpr[::step].tolist()
    if fpr_ds[-1] != 1.0:
        fpr_ds.append(1.0)
        tpr_ds.append(1.0)
        
    precomputed_metrics[name] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'classification_report': rep,
        'roc_curve': {
            'fpr': fpr_ds,
            'tpr': tpr_ds
        }
    }

# Save metrics JSON
with open("Customer_Transaction_Dashboard/data/precomputed_metrics.json", "w") as f:
    json.dump(precomputed_metrics, f, indent=4)

print("Precomputation finished successfully! Metrics saved.")
