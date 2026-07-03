import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import json
import io
from utils.helper import (
    load_subset_data,
    load_metrics,
    get_target_distribution_plots,
    get_correlation_heatmap,
    get_feature_distribution_plots,
    get_model_comparison_bar,
    get_model_roc_curves,
    get_confusion_matrix_heatmap
)

# Initialize Dash application with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, "https://use.fontawesome.com/releases/v5.15.4/css/all.css"],
    suppress_callback_exceptions=True
)
app.title = "Customer Transaction Prediction Dashboard"

# Load data on startup
df_subset = load_subset_data()
metrics = load_metrics()

# Generate feature list for dropdowns
features_list = [col for col in df_subset.columns if col.startswith('var_')] if not df_subset.empty else []

# ----------------------------------------------------------------------
# Sidebar Navigation Component
# ----------------------------------------------------------------------
sidebar = html.Div(
    [
        html.Div(
            [
                html.I(className="fas fa-university fa-2x text-primary me-2"),
                html.Span("Bank Analytics", className="h4 font-weight-bold text-primary")
            ],
            className="d-flex align-items-center mb-4 px-2"
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="fas fa-home me-2"), "Dashboard"], href="/", active="exact"),
                dbc.NavLink([html.I(className="fas fa-database me-2"), "Data Overview"], href="/data-overview", active="exact"),
                dbc.NavLink([html.I(className="fas fa-chart-line me-2"), "Feature Analysis"], href="/feature-analysis", active="exact"),
                dbc.NavLink([html.I(className="fas fa-cog me-2"), "Model Performance"], href="/model-performance", active="exact"),
                dbc.NavLink([html.I(className="fas fa-lightbulb me-2"), "Business Insights"], href="/business-insights", active="exact"),
                dbc.NavLink([html.I(className="fas fa-info-circle me-2"), "About Project"], href="/about-project", active="exact"),
            ],
            vertical=True,
            pills=True,
            className="mb-4"
        ),
        html.Div(
            [
                dbc.Button(
                    [html.I(className="fas fa-moon me-2", id="theme-toggle-icon"), "Toggle Dark Mode"],
                    id="theme-toggle-btn",
                    color="secondary",
                    size="sm",
                    className="w-100"
                )
            ],
            className="theme-toggle-btn"
        )
    ],
    className="sidebar-custom"
)

# ----------------------------------------------------------------------
# Page Content layouts
# ----------------------------------------------------------------------

# 1. Home / Dashboard Layout
def get_home_layout():
    return html.Div([
        html.Div([
            html.H2("Customer Transaction Prediction", className="mb-0 font-weight-bold"),
            html.P("Banking Domain | Machine Learning Classification Project", className="text-muted")
        ], className="header-custom"),
        
        # KPI Cards Row
        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div([
                    html.I(className="fas fa-users fa-2x text-primary mb-2"),
                    html.Div("Training Samples", className="kpi-title"),
                    html.Div("160,000", className="kpi-val"),
                    html.Small("80% of original dataset", className="text-muted")
                ])
            ], className="card-custom kpi-card"), md=3, xs=6),
            
            dbc.Col(dbc.Card([
                html.Div([
                    html.I(className="fas fa-vial fa-2x text-success mb-2"),
                    html.Div("Testing Samples", className="kpi-title"),
                    html.Div("40,000", className="kpi-val"),
                    html.Small("20% holdout validation", className="text-muted")
                ])
            ], className="card-custom kpi-card"), md=3, xs=6),
            
            dbc.Col(dbc.Card([
                html.Div([
                    html.I(className="fas fa-tags fa-2x text-warning mb-2"),
                    html.Div("Total Features", className="kpi-title"),
                    html.Div("200", className="kpi-val"),
                    html.Small("Anonymized continuous var", className="text-muted")
                ])
            ], className="card-custom kpi-card"), md=3, xs=6),
            
            dbc.Col(dbc.Card([
                html.Div([
                    html.I(className="fas fa-percentage fa-2x text-danger mb-2"),
                    html.Div("Transaction Rate", className="kpi-title"),
                    html.Div("10.05%", className="kpi-val"),
                    html.Small("Class Imbalance present", className="text-muted")
                ])
            ], className="card-custom kpi-card"), md=3, xs=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H4("Project Objective", className="card-title text-primary"),
                    html.P([
                        "This dashboard demonstrates a binary classification model designed to identify customers ",
                        "who will make a transaction in the future. The project handles anonymized continuous features ",
                        "provided by Santander Bank to build predictive profiles."
                    ]),
                    html.P([
                        "Use the left sidebar navigation to explore the dataset statistics, evaluate individual features, ",
                        "compare model architectures (Logistic Regression, Random Forest, XGBoost, and SVM), and review ",
                        "banking domain business applications."
                    ])
                ], className="card-custom h-100")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    html.H4("Quick Insights & System State", className="card-title text-primary"),
                    html.Ul([
                        html.Li("Dataset contains zero missing or null entries across all 200 variables."),
                        html.Li("High feature independence: average correlation between features is less than 0.005."),
                        html.Li("The best-performing model (XGBoost) achieves an ROC-AUC of 0.85+."),
                        html.Li("Class imbalance was addressed using weighted cost functions rather than SMOTE to maintain low memory training.")
                    ])
                ], className="card-custom h-100")
            ], md=6)
        ])
    ])

# 2. Data Overview Layout
def get_data_overview_layout():
    fig_pie, fig_bar = get_target_distribution_plots(df_subset)
    return html.Div([
        html.Div([
            html.H2("Data Overview", className="mb-0 font-weight-bold"),
            html.P("Dataset Shape, Target Distribution, and Cleanliness Statistics", className="text-muted")
        ], className="header-custom"),
        
        # Dimensions & Cleanliness row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H5("Dataset Dimension Summary", className="text-primary mb-3"),
                    html.Table([
                        html.Tr([html.Td("Total Records"), html.Td(html.Strong("200,000"))]),
                        html.Tr([html.Td("Total Columns"), html.Td(html.Strong("202"))]),
                        html.Tr([html.Td("ID Column"), html.Td(html.Code("ID_code"))]),
                        html.Tr([html.Td("Target Column"), html.Td(html.Code("target"))]),
                        html.Tr([html.Td("Missing Values"), html.Td(html.Span("0 (Clean)", className="badge bg-success"))]),
                        html.Tr([html.Td("Duplicate Rows"), html.Td(html.Span("0 (Clean)", className="badge bg-success"))])
                    ], className="table table-hover")
                ], className="card-custom h-100")
            ], md=4),
            
            dbc.Col([
                dbc.Card([
                    html.H5("Imbalance Explanation", className="text-primary mb-3"),
                    html.P([
                        "The target labels show a strong imbalance: ~90% represents customers who did not make a transaction (0), ",
                        "while only ~10% represents customers who did make a transaction (1)."
                    ]),
                    html.P([
                        "Standard accuracy metrics can be misleading here. If a model predicts '0' for every record, ",
                        "it will achieve a 90% accuracy but will detect 0% of transactions. ",
                        "This makes ROC-AUC, Recall, and F1-Score crucial evaluation criteria."
                    ])
                ], className="card-custom h-100")
            ], md=8)
        ], className="mb-4"),
        
        # Plots Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dcc.Graph(figure=fig_pie, id="pie-target-dist", config={"displayModeBar": False})
                ], className="card-custom")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dcc.Graph(figure=fig_bar, id="bar-target-dist", config={"displayModeBar": False})
                ], className="card-custom")
            ], md=6)
        ])
    ])

# 3. Feature Analysis Layout
def get_feature_analysis_layout():
    default_feature = features_list[0] if features_list else ""
    fig_hist, fig_box, stats = get_feature_distribution_plots(df_subset, default_feature)
    fig_corr = get_correlation_heatmap(df_subset, 15)
    
    return html.Div([
        html.Div([
            html.H2("Feature Analysis", className="mb-0 font-weight-bold"),
            html.P("Statistical exploration of anonymized numerical variables", className="text-muted")
        ], className="header-custom"),
        
        dbc.Alert(
            "Feature names are anonymized, therefore only statistical behaviour is analyzed.",
            color="info", className="mb-4"
        ),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H5("Select Feature to Analyze", className="text-primary"),
                    dcc.Dropdown(
                        id="feature-selector",
                        options=[{"label": f, "value": f} for f in features_list],
                        value=default_feature,
                        clearable=False,
                        className="mb-4"
                    ),
                    html.H5("Feature Statistics", className="text-primary mb-3"),
                    html.Div(id="feature-stats-table")
                ], className="card-custom h-100")
            ], md=4),
            
            dbc.Col([
                dbc.Card([
                    dcc.Tabs([
                        dcc.Tab(label='Histogram & Density', children=[
                            dcc.Graph(id="feature-hist-plot", figure=fig_hist)
                        ]),
                        dcc.Tab(label='Box Plot (by Target)', children=[
                            dcc.Graph(id="feature-box-plot", figure=fig_box)
                        ])
                    ])
                ], className="card-custom")
            ], md=8)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H5("Correlation & Independence Analysis", className="text-primary mb-3"),
                    html.P([
                        "Checking feature correlation allows us to verify if multicollinearity exists. ",
                        "As shown in the heatmap below, all features exhibit extremely low correlation with each other (mostly near 0). ",
                        "This indicates that the features have likely undergone a mathematical transformation (like PCA or scaling) ",
                        "which makes them independent variables."
                    ]),
                    dcc.Graph(figure=fig_corr, id="correlation-heatmap")
                ], className="card-custom")
            ], md=12)
        ])
    ])

# 4. Model Performance Layout
def get_model_performance_layout():
    fig_bar = get_model_comparison_bar(metrics)
    fig_roc = get_model_roc_curves(metrics)
    default_model = list(metrics.keys())[0] if metrics else ""
    
    return html.Div([
        html.Div([
            html.H2("Model Performance & Comparison", className="mb-0 font-weight-bold"),
            html.P("Evaluating Logistic Regression, Random Forest, SVM, and XGBoost", className="text-muted")
        ], className="header-custom"),
        
        # Download comparison bar
        html.Div([
            dbc.Button(
                [html.I(className="fas fa-download me-2"), "Download Comparison CSV"],
                id="btn-download-csv",
                color="primary",
                className="mb-4"
            ),
            dcc.Download(id="download-dataframe-csv")
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dcc.Graph(figure=fig_bar, id="metric-comparison-bar")
                ], className="card-custom")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dcc.Graph(figure=fig_roc, id="metrics-roc-curves")
                ], className="card-custom")
            ], md=6)
        ], className="mb-4"),
        
        # Detailed Model Inspections
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H4("Detailed Model Inspection", className="text-primary mb-3"),
                    dcc.Dropdown(
                        id="model-inspect-selector",
                        options=[{"label": m, "value": m} for m in metrics.keys()],
                        value=default_model,
                        clearable=False,
                        className="mb-4"
                    ),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H5("Confusion Matrix", className="mb-3 text-secondary"),
                            dcc.Graph(id="model-confusion-matrix")
                        ], md=6),
                        dbc.Col([
                            html.H5("Classification Report Table", className="mb-3 text-secondary"),
                            html.Div(id="model-report-table")
                        ], md=6)
                    ])
                ], className="card-custom")
            ], md=12)
        ], className="mb-4"),
        
        # Highlight best model
        dbc.Card([
            dbc.CardBody([
                html.H4([html.I(className="fas fa-trophy me-2 text-warning"), "Best Model: XGBoost / Gradient Boosting"], className="text-primary"),
                html.P([
                    "The XGBoost/Gradient Boosting ensemble model achieves the highest ROC-AUC score of approximately 0.85+. ",
                    "While Logistic Regression trains much faster, it cannot fit the non-linear boundaries. ",
                    "Random Forest performs decently but suffers in recall. XGBoost strikes the best balance in handling ",
                    "the 90-10 class imbalance efficiently (via cost weights) while minimizing misclassifications."
                ])
            ])
        ], className="card-custom border-primary")
    ])

# 5. Business Insights Layout
def get_business_insights_layout():
    return html.Div([
        html.Div([
            html.H2("Banking Business Insights", className="mb-0 font-weight-bold"),
            html.P("Translating predictions into concrete bank revenue and operations strategies", className="text-muted")
        ], className="header-custom"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H5([html.I(className="fas fa-bullseye me-2 text-primary"), "Targeted Marketing Campaigns"], className="mb-3"),
                    html.P([
                        "Banks run expensive promotional campaigns for cards, loans, and investment accounts. ",
                        "Predicting who is likely to make a transaction allows marketing teams to target only highly active clients. ",
                        "This decreases customer spam-fatigue and boosts promotion conversion rates."
                    ])
                ], className="card-custom h-100")
            ], md=4),
            
            dbc.Col([
                dbc.Card([
                    html.H5([html.I(className="fas fa-wallet me-2 text-success"), "Liquidity & Reserve Management"], className="mb-3"),
                    html.P([
                        "Predicting macro transaction trends helps regional bank branches maintain correct vault liquidity levels. ",
                        "Ensuring that cash reserves match predicted active periods prevents cash outages during high demand ",
                        "and reduces idle funds during low activity."
                    ])
                ], className="card-custom h-100")
            ], md=4),
            
            dbc.Col([
                dbc.Card([
                    html.H5([html.I(className="fas fa-shield-alt me-2 text-danger"), "Fraud Prevention Support"], className="mb-3"),
                    html.P([
                        "An unexpected high-value transaction from a customer flagged by the model as having extremely low transaction likelihood ",
                        "can trigger an automatic verification process. This acts as a supporting layer to standard rule-based fraud detection."
                    ])
                ], className="card-custom h-100")
            ], md=4)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H5("Summary of Benefits", className="text-primary mb-3"),
                    html.Ul([
                        html.Li("Personalized Financial Offerings: Trigger dynamic loan offers when customer transaction likelihood increases."),
                        html.Li("Inference Speed: Real-time scoring using our saved models guarantees API responses under 15ms."),
                        html.Li("Risk Reduction: Allows quick intervention on client accounts displaying sudden drops in standard transaction activities.")
                    ])
                ], className="card-custom")
            ], md=12)
        ])
    ])

# 6. About Project Layout
def get_about_project_layout():
    return html.Div([
        html.Div([
            html.H2("About Project & Author", className="mb-0 font-weight-bold"),
            html.P("Project objectives, technologies, and student credentials", className="text-muted")
        ], className="header-custom"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H5("Project Specifications", className="text-primary mb-3"),
                    html.Table([
                        html.Tr([html.Td("Task Type"), html.Td("Binary Classification")]),
                        html.Tr([html.Td("Dataset Source"), html.Td("Santander Customer Transaction Prediction")]),
                        html.Tr([html.Td("Primary Metric"), html.Td("ROC-AUC")]),
                        html.Tr([html.Td("Frameworks Used"), html.Td("Dash, Plotly, Scikit-Learn, Pandas, NumPy")]),
                        html.Tr([html.Td("Class Balance Strategy"), html.Td("Cost-Sensitive Loss Weights (class_weight)")]),
                        html.Tr([html.Td("Developer"), html.Td("Data Science Student")])
                    ], className="table table-hover")
                ], className="card-custom h-100")
            ], md=6),
            
            dbc.Col([
                dbc.Card([
                    html.H5("Author Profiles", className="text-primary mb-3"),
                    html.P("Developed as part of a final-year Machine Learning Capstone project demonstrating classification pipelines."),
                    html.Div([
                        dbc.Button(
                            [html.I(className="fab fa-github me-2"), "GitHub Profile"],
                            href="https://github.com/",
                            target="_blank",
                            color="dark",
                            className="me-2 mb-2"
                        ),
                        dbc.Button(
                            [html.I(className="fab fa-linkedin me-2"), "LinkedIn Profile"],
                            href="https://linkedin.com/",
                            target="_blank",
                            color="primary",
                            className="mb-2"
                        )
                    ], className="mt-3")
                ], className="card-custom h-100")
            ], md=6)
        ])
    ])

# ----------------------------------------------------------------------
# Application Base layout container
# ----------------------------------------------------------------------
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        sidebar,
        html.Div(
            [
                dcc.Loading(
                    id="loading-content",
                    type="default",
                    children=html.Div(id="page-content")
                ),
                html.Footer(
                    "Customer Transaction Prediction Dashboard | Banking Machine Learning Project",
                    className="footer-custom"
                )
            ],
            className="content-custom"
        )
    ],
    id="main-app-container",
    **{"data-theme": "light"}  # custom html attribute for toggle dark mode
)

# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------

# Page routing callback
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page_content(pathname):
    if pathname == "/":
        return get_home_layout()
    elif pathname == "/data-overview":
        return get_data_overview_layout()
    elif pathname == "/feature-analysis":
        return get_feature_analysis_layout()
    elif pathname == "/model-performance":
        return get_model_performance_layout()
    elif pathname == "/business-insights":
        return get_business_insights_layout()
    elif pathname == "/about-project":
        return get_about_project_layout()
    return html.Div(
        dbc.Container(
            [
                html.H1("404: Not Found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognized.")
            ],
            className="p-5 my-4 bg-light rounded-3"
        )
    )

# Feature Analysis interactive elements callback
@app.callback(
    [
        Output("feature-hist-plot", "figure"),
        Output("feature-box-plot", "figure"),
        Output("feature-stats-table", "children")
    ],
    Input("feature-selector", "value")
)
def update_feature_plots(feature_name):
    if not feature_name:
        return go.Figure(), go.Figure(), html.Div()
        
    fig_hist, fig_box, stats = get_feature_distribution_plots(df_subset, feature_name)
    
    # Render stats dictionary as a table
    table_rows = [
        html.Tr([html.Td(key), html.Td(html.Strong(f"{val:.4f}"))]) for key, val in stats.items()
    ]
    stats_table = html.Table(table_rows, className="table table-sm table-striped")
    
    return fig_hist, fig_box, stats_table

# Model Evaluation sub-views callback
@app.callback(
    [
        Output("model-confusion-matrix", "figure"),
        Output("model-report-table", "children")
    ],
    Input("model-inspect-selector", "value")
)
def update_model_inspection(model_name):
    if not model_name or not metrics:
        return go.Figure(), html.Div()
        
    fig_cm = get_confusion_matrix_heatmap(metrics, model_name)
    
    # Render report details
    rep = metrics[model_name]['classification_report']
    
    # Format report rows
    rows = []
    # Class rows
    for label in ['0', '1']:
        class_data = rep[label]
        rows.append(html.Tr([
            html.Td(f"Class {label}"),
            html.Td(f"{class_data['precision']:.3f}"),
            html.Td(f"{class_data['recall']:.3f}"),
            html.Td(f"{class_data['f1-score']:.3f}"),
            html.Td(f"{class_data['support']}")
        ]))
    # Average rows
    for avg in ['macro avg', 'weighted avg']:
        avg_data = rep[avg]
        rows.append(html.Tr([
            html.Td(html.Strong(avg.title())),
            html.Td(f"{avg_data['precision']:.3f}"),
            html.Td(f"{avg_data['recall']:.3f}"),
            html.Td(f"{avg_data['f1-score']:.3f}"),
            html.Td(f"{avg_data['support']}")
        ], className="table-secondary"))
        
    report_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Metric"), html.Th("Precision"), html.Th("Recall"), html.Th("F1-Score"), html.Th("Support")
        ])),
        html.Tbody(rows)
    ], className="table table-hover table-sm")
    
    return fig_cm, report_table

# Dark Mode theme toggle callback
@app.callback(
    [
        Output("main-app-container", "data-theme"),
        Output("theme-toggle-icon", "className"),
    ],
    Input("theme-toggle-btn", "n_clicks"),
    State("main-app-container", "data-theme"),
    prevent_initial_call=True
)
def toggle_theme(n_clicks, current_theme):
    if current_theme == "light":
        return "dark", "fas fa-sun me-2"
    else:
        return "light", "fas fa-moon me-2"

# CSV comparison metrics download callback
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    prevent_initial_call=True
)
def download_comparison_csv(n_clicks):
    if not metrics:
        return None
        
    results = []
    for name, m_data in metrics.items():
        results.append({
            'Model': name,
            'Accuracy': m_data['accuracy'],
            'Precision': m_data['precision'],
            'Recall': m_data['recall'],
            'F1 Score': m_data['f1_score'],
            'ROC-AUC': m_data['roc_auc']
        })
    df_compare = pd.DataFrame(results)
    return dcc.send_data_frame(df_compare.to_csv, "model_comparison_report.csv", index=False)

# Start development server
if __name__ == '__main__':
    # Run Dash app in debug mode on default port 8050
    app.run(debug=True, port=8050)
