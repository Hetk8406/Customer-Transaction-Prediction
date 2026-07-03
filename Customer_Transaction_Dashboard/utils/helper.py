import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Dynamic absolute path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_subset_data():
    """Loads the fast 10,000-row dataset for dynamic interactive plotting."""
    path = os.path.join(BASE_DIR, 'data', 'train_subset.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_metrics():
    """Loads precomputed model metrics to prevent long run times."""
    path = os.path.join(BASE_DIR, 'data', 'precomputed_metrics.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

# 1. Target Distribution Plots
def get_target_distribution_plots(df):
    if df.empty:
        return go.Figure(), go.Figure()
        
    counts = df['target'].value_counts().reset_index()
    counts.columns = ['target_name', 'count']
    counts['target_name'] = counts['target_name'].map({0: 'No Transaction (0)', 1: 'Transaction (1)'})
    
    # Pie chart
    fig_pie = px.pie(
        counts, 
        values='count', 
        names='target_name', 
        title='Target Proportion',
        color='target_name',
        color_discrete_map={'No Transaction (0)': '#6c757d', 'Transaction (1)': '#0056b3'},
        hole=0.4
    )
    fig_pie.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    # Bar chart
    fig_bar = px.bar(
        counts, 
        x='target_name', 
        y='count', 
        title='Target Count Distribution',
        color='target_name',
        color_discrete_map={'No Transaction (0)': '#6c757d', 'Transaction (1)': '#0056b3'},
        text='count'
    )
    fig_bar.update_layout(
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis_title="",
        yaxis_title="Count"
    )
    
    return fig_pie, fig_bar

# 2. Correlation Heatmap
def get_correlation_heatmap(df, num_features=15):
    if df.empty:
        return go.Figure()
    
    cols = [col for col in df.columns if col.startswith('var_')][:num_features]
    corr = df[cols].corr()
    
    fig = px.imshow(
        corr,
        text_auto=False,
        aspect="auto",
        color_continuous_scale="coolwarm",
        title=f"Correlation Heatmap (First {num_features} Features)"
    )
    fig.update_layout(margin=dict(t=50, b=20, l=20, r=20))
    return fig

# 3. Dynamic Feature Distribution Plots
def get_feature_distribution_plots(df, feature_name):
    if df.empty or feature_name not in df.columns:
        return go.Figure(), go.Figure(), {}
        
    # Stats table dictionary
    stats = {
        'Mean': round(df[feature_name].mean(), 4),
        'Median': round(df[feature_name].median(), 4),
        'Std Dev': round(df[feature_name].std(), 4),
        'Min': round(df[feature_name].min(), 4),
        'Max': round(df[feature_name].max(), 4)
    }
    
    # 1. Histogram/Density Plot grouped by Target
    fig_hist = px.histogram(
        df, 
        x=feature_name, 
        color='target',
        color_discrete_map={0: '#6c757d', 1: '#0056b3'},
        marginal="rug",
        opacity=0.6,
        barmode="overlay",
        title=f"Histogram & Density of {feature_name}"
    )
    fig_hist.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis_title=feature_name,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    # 2. Box Plot
    fig_box = px.box(
        df, 
        x='target', 
        y=feature_name, 
        color='target',
        color_discrete_map={0: '#6c757d', 1: '#0056b3'},
        points="outliers",
        title=f"Box Plot of {feature_name} by Target"
    )
    fig_box.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No Transaction (0)', 'Transaction (1)']),
        xaxis_title="",
        showlegend=False
    )
    
    return fig_hist, fig_box, stats

# 4. Model Comparison Bar Chart
def get_model_comparison_bar(metrics):
    if not metrics:
        return go.Figure()
        
    models = list(metrics.keys())
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    
    fig = go.Figure()
    
    for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        vals = [metrics[m][metric_name] for m in models]
        fig.add_trace(go.Bar(
            name=metric_name.replace('_', ' ').title().replace('Roc', 'ROC'),
            x=models,
            y=vals,
            text=[f"{v:.3f}" for v in vals],
            textposition='auto'
        ))
        
    fig.update_layout(
        barmode='group',
        title='Model Metrics Comparison',
        yaxis=dict(range=[0, 1.1]),
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig

# 5. Precomputed ROC Curves
def get_model_roc_curves(metrics):
    fig = go.Figure()
    
    colors = {
        'Logistic Regression': '#17a2b8',
        'Random Forest': '#fd7e14',
        'XGBoost/GB': '#0056b3',
        'Linear SVM': '#6f42c1'
    }
    
    for name, m_data in metrics.items():
        fpr = m_data['roc_curve']['fpr']
        tpr = m_data['roc_curve']['tpr']
        auc = m_data['roc_auc']
        
        fig.add_trace(go.Scatter(
            x=fpr, 
            y=tpr, 
            mode='lines',
            name=f"{name} (AUC = {auc:.3f})",
            line=dict(color=colors.get(name, '#6c757d'), width=2)
        ))
        
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Random Guess'
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig

# 6. Confusion Matrix Heatmap
def get_confusion_matrix_heatmap(metrics, model_name):
    if not metrics or model_name not in metrics:
        return go.Figure()
        
    cm = metrics[model_name]['confusion_matrix']
    z = [
        [cm['tn'], cm['fp']],
        [cm['fn'], cm['tp']]
    ]
    
    x = ['Predicted 0', 'Predicted 1']
    y = ['Actual 0', 'Actual 1']
    
    fig = px.imshow(
        z,
        x=x,
        y=y,
        color_continuous_scale='Blues',
        text_auto=True,
        title=f'Confusion Matrix: {model_name}'
    )
    fig.update_layout(margin=dict(t=50, b=20, l=20, r=20))
    return fig
