"""
ML Classification Playground - Streamlit Application
A stunning, modern interactive dashboard for exploring machine learning classification models.

GitHub Repository: https://github.com/2025aa05482-bits/ML-assignment2
"""

#Test Apply
# GitHub repository URL
GITHUB_REPO_URL = "https://github.com/2025aa05482-bits/ML-assignment2"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time
from pathlib import Path

# Try to import SMOTE for advanced class imbalance handling
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Import our models
from model.logistic import LogisticRegressionModel
from model.dt import DecisionTreeModel
from model.knn import KNNModel
from model.nb import NaiveBayesModel
from model.rf import RandomForestModel
from model.xgb import XGBoostModel

# Page configuration
st.set_page_config(
    page_title="ML Classification Playground",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-modern CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg-dark: #0f0f14;
        --bg-darker: #08080c;
        --bg-card: rgba(22, 22, 32, 0.6);
        --bg-card-hover: rgba(32, 32, 48, 0.8);
        --accent-violet: #7c3aed;
        --accent-fuchsia: #d946ef;
        --accent-cyan: #22d3ee;
        --accent-emerald: #34d399;
        --accent-amber: #fbbf24;
        --accent-rose: #fb7185;
        --text-white: #fafafa;
        --text-gray: #a1a1aa;
        --text-muted: #71717a;
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-glow: rgba(124, 58, 237, 0.5);
        --gradient-1: linear-gradient(135deg, #7c3aed, #d946ef);
        --gradient-2: linear-gradient(135deg, #22d3ee, #34d399);
        --gradient-3: linear-gradient(135deg, #fb7185, #fbbf24);
    }
    
    /* Global Styles */
    .stApp {
        background: var(--bg-dark);
        background-image: 
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(124, 58, 237, 0.15), transparent),
            radial-gradient(ellipse 60% 40% at 100% 100%, rgba(217, 70, 239, 0.1), transparent),
            radial-gradient(ellipse 40% 30% at 0% 80%, rgba(34, 211, 238, 0.08), transparent);
        min-height: 100vh;
    }
    
    /* Hide Streamlit Elements */
    footer, header {visibility: hidden;}
    
    /* ============ FORCE SIDEBAR VISIBILITY ============ */
    [data-testid="stSidebar"] {
        visibility: visible !important;
        display: block !important;
        position: relative !important;
        width: 21rem !important;
        min-width: 21rem !important;
        max-width: 21rem !important;
    }
    
    /* Ensure sidebar content is visible */
    [data-testid="stSidebar"] > div {
        visibility: visible !important;
        display: block !important;
    }
    
    /* Make sidebar toggle button visible */
    button[kind="header"] {
        visibility: visible !important;
        display: block !important;
    }
    
    /* ============ HERO SECTION ============ */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        position: relative;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 50px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--accent-violet);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1.25rem;
    }
    
    .hero-badge::before {
        content: '';
        width: 6px;
        height: 6px;
        background: var(--accent-violet);
        border-radius: 50%;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.5); }
    }
    
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #e4e4e7 30%, var(--accent-violet) 70%, var(--accent-fuchsia) 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-flow 6s ease infinite;
    }
    
    @keyframes gradient-flow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        color: var(--text-gray);
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* ============ STATS BAR ============ */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 3rem;
        padding: 1.5rem 2rem;
        margin: 1.5rem auto;
        max-width: 700px;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        backdrop-filter: blur(20px);
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text-white);
        line-height: 1.2;
    }
    
    .stat-label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.25rem;
    }
    
    /* ============ SECTION HEADERS ============ */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-white);
        margin: 2.5rem 0 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-title::before {
        content: '';
        width: 4px;
        height: 24px;
        background: var(--gradient-1);
        border-radius: 4px;
    }
    
    /* ============ METRIC CARDS ============ */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-1);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-glow);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px -20px rgba(124, 58, 237, 0.3);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.75rem;
        display: block;
        filter: drop-shadow(0 4px 8px rgba(124, 58, 237, 0.3));
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .metric-value.accuracy { color: var(--accent-emerald); }
    .metric-value.precision { color: var(--accent-violet); }
    .metric-value.recall { color: var(--accent-amber); }
    .metric-value.f1 { color: var(--accent-cyan); }
    .metric-value.neutral { color: var(--text-white); }
    
    .metric-label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    /* ============ GLASS CARDS ============ */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .glass-card-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-white);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* ============ SIDEBAR STYLES ============ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 15, 20, 0.98) 0%, rgba(8, 8, 12, 0.98) 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
        visibility: visible !important;
        display: block !important;
        z-index: 999 !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1rem;
    }
    
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    
    .sidebar-section-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.75rem;
        padding-left: 0.25rem;
    }
    
    .model-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(217, 70, 239, 0.05));
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0;
    }
    
    .model-card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-white);
        margin-bottom: 0.5rem;
    }
    
    .model-card-desc {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.8rem;
        color: var(--text-gray);
        line-height: 1.5;
    }
    
    /* ============ BUTTONS ============ */
    .stButton > button {
        background: var(--gradient-1);
        color: white;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px -4px rgba(124, 58, 237, 0.5);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px -4px rgba(124, 58, 237, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* ============ FORM ELEMENTS ============ */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(22, 22, 32, 0.8);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stSlider > div > div > div {
        background: var(--gradient-1);
    }
    
    /* ============ EXPANDER ============ */
    .streamlit-expanderHeader {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 500;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
    }
    
    /* ============ TABS ============ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        color: var(--text-gray);
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.75rem 1.25rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-card-hover);
        color: var(--text-white);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-1);
        border-color: transparent;
        color: white;
    }
    
    /* ============ DATA TABLES ============ */
    .stDataFrame {
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* ============ EMPTY STATE ============ */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: var(--bg-card);
        border: 2px dashed var(--border-subtle);
        border-radius: 24px;
        margin: 2rem 0;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        display: block;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .empty-state-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-white);
        margin-bottom: 0.75rem;
    }
    
    .empty-state-desc {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1rem;
        color: var(--text-muted);
        max-width: 400px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* ============ RESULT BANNER ============ */
    .result-banner {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.1), rgba(52, 211, 153, 0.1));
        border: 1px solid rgba(34, 211, 238, 0.3);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .result-banner-icon {
        font-size: 1.75rem;
    }
    
    .result-banner-text {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.95rem;
        color: var(--text-white);
    }
    
    .result-banner-highlight {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: var(--accent-cyan);
    }
    
    /* ============ SCROLLBAR ============ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(124, 58, 237, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(124, 58, 237, 0.5);
    }
    
    /* ============ DIVIDER ============ */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
        margin: 2rem 0;
    }
    
    /* ============ TOOLTIP ============ */
    .tooltip-card {
        background: rgba(0, 0, 0, 0.9);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.8rem;
        color: var(--text-gray);
    }
    
    /* ============ FEATURE TAG ============ */
    .feature-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.75rem;
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 6px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--accent-violet);
        margin: 0.25rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.5rem; }
        .metric-grid { grid-template-columns: repeat(2, 1fr); }
        .stats-bar { flex-wrap: wrap; gap: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)


# Model registry
MODELS = {
    'Logistic Regression': LogisticRegressionModel,
    'Decision Tree': DecisionTreeModel,
    'K-Nearest Neighbors': KNNModel,
    'Naive Bayes': NaiveBayesModel,
    'Random Forest': RandomForestModel,
    'XGBoost': XGBoostModel
}

MODEL_ICONS = {
    'Logistic Regression': '📈',
    'Decision Tree': '🌳',
    'K-Nearest Neighbors': '🎯',
    'Naive Bayes': '🎲',
    'Random Forest': '🌲',
    'XGBoost': '⚡'
}

MODEL_DESCRIPTIONS = {
    'Logistic Regression': 'Linear classifier using sigmoid function for probability estimation. Best for linearly separable data.',
    'Decision Tree': 'Tree-based model making decisions through feature threshold splits. Highly interpretable.',
    'K-Nearest Neighbors': 'Instance-based learning using proximity to k nearest neighbors for classification.',
    'Naive Bayes': 'Probabilistic classifier applying Bayes theorem with feature independence assumption.',
    'Random Forest': 'Ensemble of decision trees with bagging and random feature selection for robustness.',
    'XGBoost': 'Advanced gradient boosting with regularization. Often wins ML competitions.'
}

# Hyperparameter grids for auto-tuning (focused on accuracy)
PARAM_GRIDS = {
    'Logistic Regression': {
        'C': [0.1, 0.5, 1.0, 5.0, 10.0],
        'max_iter': [1000, 2000],
        'solver': ['lbfgs', 'saga'],
        'class_weight': ['balanced']
    },
    'Decision Tree': {
        'max_depth': [8, 12, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced']
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [5, 11, 15, 21, 25],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Naive Bayes': {
        'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]
    },
    'Random Forest': {
        'n_estimators': [200, 300, 400],
        'max_depth': [12, 15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    },
    'XGBoost': {
        'n_estimators': [200, 300, 400],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1, 3, 5]
    }
}

# Sklearn estimators for GridSearchCV
SKLEARN_MODELS = {
    'Logistic Regression': LogisticRegression,
    'Decision Tree': DecisionTreeClassifier,
    'K-Nearest Neighbors': KNeighborsClassifier,
    'Naive Bayes': GaussianNB,
    'Random Forest': RandomForestClassifier,
    'XGBoost': XGBClassifier
}


def load_uploaded_file(uploaded_file):
    """
    Load DataFrame from uploaded file. Supports .csv, .data, and .test.
    .data and .test are typically whitespace-delimited (space/tab).
    """
    name = (uploaded_file.name or "").lower()
    # Reset position in case it was read before
    uploaded_file.seek(0)
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".data") or name.endswith(".test"):
        # Whitespace-delimited (space/tab); .data/.test often have no header
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, sep=r"\s+", header=0)
            # If column names look numeric (first row was data), re-read without header
            first_col = str(df.columns[0]) if len(df.columns) else ""
            try:
                float(first_col)
                has_numeric_header = True
            except (ValueError, TypeError):
                has_numeric_header = False
            if has_numeric_header or (df.shape[1] >= 1 and pd.api.types.is_numeric_dtype(df.iloc[:, 0])):
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
                df.columns = [f"feature_{i}" for i in range(df.shape[1] - 1)] + ["target"]
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
            df.columns = [f"feature_{i}" for i in range(df.shape[1] - 1)] + ["target"]
        return df
    # Default: try CSV
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def auto_detect_target_column(df):
    """
    Automatically detect the target column from a DataFrame.
    Priority: common target names > last column > first column.
    """
    cols = df.columns.tolist()
    if len(cols) == 0:
        return None
    # Common target column names (case-insensitive)
    target_names = [
        'target', 'Target', 'TARGET',
        'label', 'Label', 'LABEL',
        'class', 'Class', 'CLASS',
        'y', 'Y', 'outcome', 'Outcome',
        'result', 'Result', 'dependent'
    ]
    for name in target_names:
        if name in cols:
            return name
    # ID-like columns to avoid using as target
    skip_names = ['id', 'ID', 'Id', 'index', 'Index']
    # Prefer last column (many datasets have target as last)
    for c in reversed(cols):
        if c not in skip_names:
            return c
    return cols[-1]


def auto_tune_model(model_name, X_train, y_train, cv=5):
    """
    Perform GridSearchCV to find the best hyperparameters.
    Returns the best parameters and best cross-validation score.
    """
    model_class = SKLEARN_MODELS[model_name]
    param_grid = PARAM_GRIDS[model_name]
    
    # Create base model
    if model_name == 'XGBoost':
        base_model = model_class(eval_metric='mlogloss', verbosity=0, random_state=42)
    elif model_name in ['Logistic Regression', 'Decision Tree', 'Random Forest']:
        base_model = model_class(random_state=42)
    else:
        base_model = model_class()
    
    # Perform grid search
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_


def _is_credit_like_dataset(df):
    """Detect if dataset looks like credit/loan (has Total_income, Age, etc.)."""
    credit_markers = ['Total_income', 'Age', 'Years_employed', 'Num_family']
    return sum(1 for c in credit_markers if c in df.columns) >= 2


def engineer_features(df):
    """
    Create engineered features to improve model accuracy.
    Dataset-adaptive: rich engineering for credit-like data, light for others (e.g. wine)
    to avoid overfitting and improve accuracy.
    """
    df_eng = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    is_credit = _is_credit_like_dataset(df)

    if is_credit:
        # ---------- Credit / loan style dataset ----------
        if 'Total_income' in df.columns and 'Num_family' in df.columns:
            df_eng['Income_per_family_member'] = df['Total_income'] / (df['Num_family'] + 1)
            df_eng['Income_per_family_log'] = np.log1p(df_eng['Income_per_family_member'])
        if 'Total_income' in df.columns and 'Age' in df.columns:
            df_eng['Income_to_age_ratio'] = df['Total_income'] / (df['Age'] + 1)
        if 'Years_employed' in df.columns and 'Age' in df.columns:
            df_eng['Employment_ratio'] = df['Years_employed'] / (df['Age'] + 1)
            df_eng['Career_start_age'] = np.clip(df['Age'] - df['Years_employed'], 0, 100)
        if 'Account_length' in df.columns and 'Age' in df.columns:
            df_eng['Account_to_age_ratio'] = df['Account_length'] / (df['Age'] + 1)
        binary_cols = ['Own_car', 'Own_property', 'Work_phone', 'Phone', 'Email']
        existing_binary = [c for c in binary_cols if c in df.columns]
        if len(existing_binary) >= 2:
            df_eng['Total_assets'] = df[existing_binary].sum(axis=1)
        if 'Num_children' in df.columns and 'Num_family' in df.columns:
            df_eng['Children_ratio'] = df['Num_children'] / (df['Num_family'] + 1)
        if 'Total_income' in df.columns:
            income_mean, income_std = df['Total_income'].mean(), df['Total_income'].std()
            if income_std > 0:
                df_eng['Income_category'] = pd.cut(
                    df['Total_income'],
                    bins=[-np.inf, income_mean - income_std, income_mean, income_mean + income_std, np.inf],
                    labels=[0, 1, 2, 3]
                ).astype(float)
            df_eng['Income_log'] = np.log1p(df['Total_income'].clip(lower=0))
        if 'Age' in df.columns:
            df_eng['Age_category'] = pd.cut(
                df['Age'],
                bins=[0, 25, 35, 45, 55, 65, np.inf],
                labels=[0, 1, 2, 3, 4, 5]
            ).astype(float)
    else:
        # ---------- Generic / wine-style: light engineering to avoid overfitting ----------
        # Only add a few ratio features if columns suggest it (e.g. wine)
        if 'fixed acidity' in df.columns and 'volatile acidity' in df.columns:
            df_eng['acidity_ratio'] = df['volatile acidity'] / (df['fixed acidity'] + 1e-8)
        if 'alcohol' in df.columns and 'density' in df.columns:
            df_eng['alcohol_density'] = df['alcohol'] * (1 - df['density'])

    # Remove infinite and NaN
    df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
    for c in df_eng.columns:
        if df_eng[c].isna().any():
            df_eng[c] = df_eng[c].fillna(df_eng[c].median())
    return df_eng


def check_class_imbalance(y):
    """Check if there's significant class imbalance."""
    value_counts = pd.Series(y).value_counts()
    if len(value_counts) < 2:
        return False, 1.0
    ratio = value_counts.iloc[0] / value_counts.iloc[1]
    return ratio > 3.0, ratio




def create_confusion_matrix_plot(cm, class_names=None):
    """Create a beautiful confusion matrix heatmap."""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale=[
            [0, '#0f0f14'],
            [0.25, '#3b1d8f'],
            [0.5, '#7c3aed'],
            [0.75, '#c026d3'],
            [1, '#f472b6']
        ],
        text=cm,
        texttemplate='<b>%{text}</b>',
        textfont={'size': 18, 'color': 'white', 'family': 'JetBrains Mono'},
        hoverongaps=False,
        showscale=False
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Confusion Matrix</b>',
            font=dict(size=16, color='#fafafa', family='Space Grotesk'),
            x=0.5
        ),
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a1a1aa', family='Plus Jakarta Sans'),
        height=380,
        margin=dict(l=60, r=20, t=60, b=60),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def create_feature_importance_plot(importances, feature_names):
    """Create a horizontal bar chart for feature importances."""
    importances = np.asarray(importances)
    # Ensure same length (e.g. after feature selection, model returns only selected feature importances)
    n = min(len(importances), len(feature_names))
    if n == 0:
        return go.Figure().update_layout(title="No feature importance available")
    importances = importances[:n]
    feature_names = feature_names[:n] if isinstance(feature_names, list) else list(feature_names)[:n]
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(10)
    
    colors = px.colors.sample_colorscale('Viridis', [i/len(df) for i in range(len(df))])
    
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f'{v:.3f}' for v in df['Importance']],
        textposition='outside',
        textfont=dict(color='#a1a1aa', size=11, family='JetBrains Mono')
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Top Feature Importance</b>',
            font=dict(size=16, color='#fafafa', family='Space Grotesk'),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a1a1aa', family='Plus Jakarta Sans'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            title='',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            title='',
            tickfont=dict(size=11)
        ),
        height=400,
        margin=dict(l=20, r=80, t=60, b=40),
        showlegend=False
    )
    
    return fig


def create_metrics_radar_plot(metrics_dict):
    """Create a beautiful radar chart for metrics."""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        metrics_dict['accuracy'],
        metrics_dict['precision'],
        metrics_dict['recall'],
        metrics_dict['f1_score']
    ]
    values.append(values[0])
    
    fig = go.Figure()
    
    # Add the filled area
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(124, 58, 237, 0.25)',
        line=dict(color='#7c3aed', width=3),
        marker=dict(size=10, color='#d946ef', symbol='circle'),
        name='Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='rgba(255,255,255,0.08)',
                linecolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#71717a', size=10),
                tickvals=[0.25, 0.5, 0.75, 1.0]
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.08)',
                linecolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#a1a1aa', size=12, family='Plus Jakarta Sans')
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a1a1aa'),
        showlegend=False,
        height=350,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig


def create_distribution_plot(df, target):
    """Create distribution plots for features and classes."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Feature Distribution', 'Class Balance'),
        horizontal_spacing=0.12
    )
    
    # Feature distributions (first 3 features)
    colors = ['#7c3aed', '#22d3ee', '#34d399', '#fbbf24']
    for i, col in enumerate(df.columns[:min(4, len(df.columns))]):
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col[:15] + '...' if len(col) > 15 else col,
                opacity=0.7,
                marker_color=colors[i % len(colors)]
            ),
            row=1, col=1
        )
    
    # Class distribution
    target_counts = target.value_counts().sort_index()
    class_colors = ['#7c3aed', '#d946ef', '#22d3ee', '#34d399', '#fbbf24'][:len(target_counts)]
    
    fig.add_trace(
        go.Bar(
            x=[f'Class {i}' for i in target_counts.index],
            y=target_counts.values,
            marker=dict(
                color=class_colors,
                line=dict(width=0)
            ),
            text=target_counts.values,
            textposition='outside',
            textfont=dict(color='#a1a1aa', family='JetBrains Mono')
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a1a1aa', family='Plus Jakarta Sans'),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.25,
            font=dict(size=10)
        ),
        height=350,
        margin=dict(l=40, r=40, t=60, b=80)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', zeroline=False)
    
    # Update subplot title styling
    fig.update_annotations(font=dict(color='#fafafa', family='Space Grotesk', size=14))
    
    return fig


def create_correlation_heatmap(df):
    """Create a correlation matrix heatmap."""
    # Select only numeric columns and limit to first 8 features
    numeric_df = df.select_dtypes(include=[np.number]).iloc[:, :8]
    corr_matrix = numeric_df.corr()
    
    # Shorten feature names for display
    short_names = [name[:10] + '..' if len(name) > 12 else name for name in corr_matrix.columns]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=short_names,
        y=short_names,
        colorscale=[
            [0, '#0f0f14'],
            [0.5, '#7c3aed'],
            [1, '#d946ef']
        ],
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={'size': 9, 'color': 'white'},
        hoverongaps=False,
        showscale=True,
        colorbar=dict(
            title='',
            tickfont=dict(color='#a1a1aa'),
            thickness=15
        )
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Feature Correlation</b>',
            font=dict(size=16, color='#fafafa', family='Space Grotesk'),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a1a1aa', family='Plus Jakarta Sans', size=10),
        height=380,
        margin=dict(l=80, r=40, t=60, b=80),
        xaxis=dict(tickangle=45)
    )
    
    return fig


def render_metric_card(icon, value, label, color_class="neutral"):
    """Render a styled metric card."""
    if isinstance(value, float) and value <= 1:
        value_str = f"{value*100:.2f}%"
    else:
        value_str = str(value)
    
    st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">{icon}</span>
            <div class="metric-value {color_class}">{value_str}</div>
            <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # ============ HERO SECTION ============
    st.markdown("""
        <div class="hero-container">
            <div class="hero-badge">Assignment 2 • Machine Learning</div>
            <h1 class="hero-title">ML Classification Playground</h1>
                <p class="hero-subtitle">
                Upload your dataset, train, evaluate, and compare machine learning classification models 
                with an intuitive interactive interface
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # ============ SIDEBAR ============
    with st.sidebar:
        # Control Panel Header - Always visible
        st.markdown("""
            <div style="text-align: center; padding: 0.5rem 0 1.5rem;">
                <span style="font-size: 2rem;">🎯</span>
                <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; 
                          font-weight: 600; color: #fafafa; margin: 0.5rem 0 0;">
                    Control Panel
                </h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Data Configuration
        st.markdown('<p class="sidebar-section-title">📁 Data Source</p>', unsafe_allow_html=True)
        
        # 1. Download button (always shown first, before browse)
        _adult_data_path = Path(__file__).parent / "adult.data"
        if not _adult_data_path.exists():
            _adult_data_path = Path("adult.data")
        if _adult_data_path.exists():
            _adult_data_bytes = _adult_data_path.read_bytes()
            st.download_button(
                label="📥 Download adult.data",
                data=_adult_data_bytes,
                file_name="adult.data",
                mime="text/plain",
                help="Click to download the UCI Adult sample dataset (adult.data)"
            )
        else:
            st.caption("Sample dataset (adult.data) not found in app folder.")
        
        # 2. Browse / Upload (shown after download)
        df = None
        target = None
        
        uploaded_file = st.file_uploader(
            "Browse / Upload your data file",
            type=['csv', 'data', 'test'],
            help="Upload a CSV, .data, or .test file with features and target column"
        )
        
        if uploaded_file is not None:
            df = load_uploaded_file(uploaded_file)
            # Drop ID-like columns first so they are never used as features
            id_cols = [c for c in df.columns if c in ('ID', 'Id', 'id', 'index', 'Index')]
            if id_cols:
                df = df.drop(columns=id_cols)
            target_col = auto_detect_target_column(df)
            if target_col is None:
                target_col = df.columns[-1]
            target = df[target_col]
            df = df.drop(columns=[target_col])
            st.success(f"Target column set automatically: **{target_col}**")
            
            # Encode categorical target
            if target.dtype == 'object':
                le = LabelEncoder()
                target = pd.Series(le.fit_transform(target), name=target.name)
            
            # Handle continuous target
            unique_ratio = len(target.unique()) / len(target)
            is_continuous = target.dtype in ['float64', 'float32'] and unique_ratio > 0.1
            
            if is_continuous:
                st.warning(f"Target appears continuous ({len(target.unique())} unique values)")
                cont_handling = st.radio(
                    "Handle continuous target",
                    ["Bin into classes", "Round to integers"],
                    label_visibility="collapsed"
                )
                
                if cont_handling == "Bin into classes":
                    n_bins = st.slider("Number of classes", 2, 10, 3)
                    target = pd.qcut(target, q=n_bins, labels=False, duplicates='drop')
                else:
                    target = target.round().astype(int)
                    le = LabelEncoder()
                    target = pd.Series(le.fit_transform(target), name=target.name)
            
            # Handle categorical features with better encoding
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                st.info(f"Found {len(categorical_cols)} categorical column(s)")
                cat_handling = st.radio(
                    "Handle categorical",
                    ["Label Encode", "One-Hot Encode"],  # "Drop" commented out
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                if cat_handling == "Label Encode":
                    for col in categorical_cols:
                        le_feat = LabelEncoder()
                        df[col] = le_feat.fit_transform(df[col].astype(str))
                elif cat_handling == "One-Hot Encode":
                    # One-hot encode for low cardinality features, label encode for high cardinality
                    for col in categorical_cols:
                        n_unique = df[col].nunique()
                        if n_unique <= 10:  # One-hot encode if <= 10 unique values
                            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                        else:  # Label encode if > 10 unique values
                            le_feat = LabelEncoder()
                            df[col] = le_feat.fit_transform(df[col].astype(str))
                # else:  # Drop option commented out
                #     df = df.drop(columns=categorical_cols)
            
            # Handle missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                st.warning(f"Found {missing_count} missing value(s)")
                nan_handling = st.radio(
                    "Handle missing",
                    ["Fill", "Drop rows"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                if nan_handling == "Fill":
                    for col in df.columns:
                        if df[col].dtype in ['float64', 'int64']:
                            df[col] = df[col].fillna(df[col].mean())
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
                    if target.isnull().any():
                        target = target.fillna(target.mode()[0] if not target.mode().empty else 0)
                else:
                    mask = ~(df.isnull().any(axis=1) | target.isnull())
                    df = df[mask].reset_index(drop=True)
                    target = target[mask].reset_index(drop=True)
            
            # Ensure target classes are 0, 1, 2, ... (required by XGBoost and some sklearn models)
            uniq = np.unique(target.values)
            if len(uniq) > 0:
                need_encode = (uniq != np.arange(len(uniq))).any()
                if need_encode:
                    le_final = LabelEncoder()
                    target = pd.Series(le_final.fit_transform(target.values), name=target.name)
            
            # Check class imbalance
            is_imbalanced, imbalance_ratio = check_class_imbalance(target)
            if is_imbalanced:
                st.warning(f"⚠️ Class imbalance detected ({imbalance_ratio:.1f}:1 ratio). Using balanced class weights.")
                st.session_state['class_imbalance'] = True
                
                # SMOTE option
                if SMOTE_AVAILABLE:
                    use_smote = st.checkbox(
                        "🔄 Use SMOTE Oversampling",
                        value=True,
                        help="Synthetic Minority Over-sampling Technique to balance classes"
                    )
                    st.session_state['use_smote'] = use_smote
                else:
                    st.info("💡 Install `imbalanced-learn` for SMOTE support: `pip install imbalanced-learn`")
                    st.session_state['use_smote'] = False
            else:
                st.session_state['class_imbalance'] = False
                st.session_state['use_smote'] = False
            
            # Feature Engineering
            use_feature_eng = st.checkbox(
                "🔬 Enable Feature Engineering",
                value=True,
                help="Create new features to improve model accuracy"
            )
            
            if use_feature_eng:
                original_cols = len(df.columns)
                df = engineer_features(df)
                new_cols = len(df.columns) - original_cols
                if new_cols > 0:
                    st.success(f"✅ Created {new_cols} new engineered features")
            
            # ID columns already dropped at load time
        
        st.markdown("---")
        
        # Model Selection
        st.markdown('<p class="sidebar-section-title">🤖 Model Selection</p>', unsafe_allow_html=True)
        
        selected_model = st.selectbox(
            "Choose Algorithm",
            list(MODELS.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown(f"""
            <div class="model-card">
                <div class="model-card-title">{MODEL_ICONS[selected_model]} {selected_model}</div>
                <div class="model-card-desc">{MODEL_DESCRIPTIONS[selected_model]}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Get model class and set default hyperparameters
        model_class = MODELS[selected_model]
        param_info = model_class.get_param_info()
        
        # Use default hyperparameters from model
        hyperparams = {}
        for param_name, info in param_info.items():
            hyperparams[param_name] = info['default']
        
        st.markdown("---")
        
        # Set default training settings (no UI controls)
        test_size = 0.2
        scale_method = "StandardScaler"
        use_feature_selection = True
        n_features = 20
        random_state = 42
    
    # ============ MAIN CONTENT ============
    if df is not None and target is not None:
        
        # Stats Bar
        st.markdown(f"""
            <div class="stats-bar">
                <div class="stat-item">
                    <div class="stat-number">{len(df):,}</div>
                    <div class="stat-label">Samples</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len(df.columns)}</div>
                    <div class="stat-label">Features</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len(target.unique())}</div>
                    <div class="stat-label">Classes</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{df.isnull().sum().sum()}</div>
                    <div class="stat-label">Missing</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🚀 Train Model", "📈 Results"])
        
        with tab1:
            st.markdown('<h2 class="section-title">Data Overview</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            if df.shape[1] == 0:
                st.warning("No feature columns remaining. Please upload a file with at least one feature column (besides the target).")
            else:
                with col1:
                    # Distribution plot
                    fig_dist = create_distribution_plot(df, target)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Correlation heatmap
                    fig_corr = create_correlation_heatmap(df)
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            # Data preview
            with st.expander("🔍 Preview Dataset", expanded=False):
                if df.shape[1] == 0:
                    st.warning("No features to display.")
                else:
                    st.dataframe(
                        df.head(10).style.format(precision=4),
                        use_container_width=True,
                        height=300
                    )
            
            # Feature statistics
            with st.expander("📊 Feature Statistics", expanded=False):
                if df.shape[1] == 0:
                    st.warning("No features to describe.")
                else:
                    st.dataframe(
                        df.describe().T.style.format(precision=4),
                        use_container_width=True
                    )
        
        with tab2:
            st.markdown('<h2 class="section-title">Model Training</h2>', unsafe_allow_html=True)
            
            # Model summary card
            st.markdown(f"""
                <div class="glass-card">
                    <div class="glass-card-title">{MODEL_ICONS[selected_model]} Selected Configuration</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
                        <span class="feature-tag">Model: {selected_model}</span>
                        <span class="feature-tag">Test Size: {test_size*100:.0f}%</span>
                        <span class="feature-tag">Scaling: {scale_method}</span>
                        <span class="feature-tag">Feature Selection: {'Yes' if use_feature_selection else 'No'}</span>
                        <span class="feature-tag">Random State: {random_state}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Training mode selection (default Auto-Tune for better accuracy)
            training_mode = st.radio(
                "Training Mode",
                ["🚀 Auto-Tune (Find best parameters)", "Manual (Use default parameters)"],
                horizontal=True,
                help="Auto-Tune uses GridSearchCV to find optimal hyperparameters for better accuracy"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if "Auto-Tune" in training_mode:
                    train_button = st.button(
                        f"🔧 Auto-Tune & Train {selected_model}",
                        use_container_width=True,
                        type="primary"
                    )
                else:
                    train_button = st.button(
                        f"🎯 Train {selected_model}",
                        use_container_width=True,
                        type="primary"
                    )
            
            if train_button:
                if df.shape[1] == 0:
                    st.error("No features to train on. Please upload a file with at least one feature column (besides the target).")
                else:
                    spinner_text = "Auto-tuning model..." if "Auto-Tune" in training_mode else "Training model..."
                    with st.spinner(spinner_text):
                        progress_bar = st.progress(0)
                        
                        # Prepare data
                        progress_bar.progress(10, "Splitting data...")
                        y_vals = target.values
                        uniq, counts = np.unique(y_vals, return_counts=True)
                        can_stratify = len(uniq) > 1 and (counts >= 2).all()
                        X_train, X_test, y_train, y_test = train_test_split(
                            df.values, y_vals,
                            test_size=test_size,
                            random_state=random_state,
                            stratify=y_vals if can_stratify else None
                        )
                        
                        # Apply SMOTE for imbalanced data if available and needed
                        use_smote = st.session_state.get('use_smote', False) and st.session_state.get('class_imbalance', False)
                        if use_smote and SMOTE_AVAILABLE:
                            progress_bar.progress(15, "Applying SMOTE oversampling...")
                            try:
                                smote = SMOTE(random_state=random_state)
                                X_train, y_train = smote.fit_resample(X_train, y_train)
                            except Exception as e:
                                st.warning(f"SMOTE failed: {str(e)}. Continuing without oversampling.")
                        
                        # Feature selection: adaptive k to avoid overfitting on small data
                        feature_selector = None
                        n_samples, n_cols = X_train.shape[0], X_train.shape[1]
                        k_select = min(n_features, n_cols, max(5, n_samples // 50))
                        if use_feature_selection and n_cols > 5 and k_select < n_cols:
                            progress_bar.progress(20, "Selecting best features...")
                            try:
                                feature_selector = SelectKBest(
                                    score_func=mutual_info_classif,
                                    k=k_select
                                )
                                X_train = feature_selector.fit_transform(X_train, y_train)
                                X_test = feature_selector.transform(X_test)
                                mask = feature_selector.get_support()
                                st.session_state['selected_features'] = mask
                                st.session_state['selected_feature_names'] = np.array(df.columns)[mask].tolist()
                            except Exception as e:
                                st.warning(f"Feature selection failed: {str(e)}. Continuing without selection.")
                        
                        # Scale features (skip if no features remain)
                        scaler = None
                        if X_train.shape[1] == 0:
                            st.error("No features remaining after preprocessing. Please use a dataset with at least one feature column.")
                        elif scale_method == "StandardScaler":
                            progress_bar.progress(25, "Scaling features (StandardScaler)...")
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                        elif scale_method == "RobustScaler":
                            progress_bar.progress(25, "Scaling features (RobustScaler)...")
                            scaler = RobustScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                        elif scale_method == "MinMaxScaler":
                            progress_bar.progress(25, "Scaling features (MinMaxScaler)...")
                            scaler = MinMaxScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                        
                        if X_train.shape[1] == 0:
                            progress_bar.empty()
                        else:
                            best_params = None
                            cv_score = None
                            
                            if "Auto-Tune" in training_mode:
                                # Auto-tune mode: find best hyperparameters
                                progress_bar.progress(30, "Finding best hyperparameters (this may take a while)...")
                                
                                try:
                                    best_params, cv_score, best_estimator = auto_tune_model(
                                        selected_model, X_train, y_train, cv=5
                                    )
                                    
                                    progress_bar.progress(70, "Training final model with best params...")
                                    
                                    # Create our wrapper model with best params
                                    wrapper_params = {}
                                    param_info = model_class.get_param_info()
                                    for key, value in best_params.items():
                                        if key in param_info:
                                            wrapper_params[key] = value
                                    
                                    model = model_class(**wrapper_params) if wrapper_params else model_class()
                                    
                                    start_time = time.time()
                                    model.train(X_train, y_train)
                                    training_time = time.time() - start_time
                                    
                                except Exception as e:
                                    st.error(f"Auto-tune failed: {str(e)}. Falling back to default parameters.")
                                    model = model_class(**hyperparams)
                                    start_time = time.time()
                                    model.train(X_train, y_train)
                                    training_time = time.time() - start_time
                            else:
                                # Manual mode: use default parameters
                                progress_bar.progress(50, "Training model...")
                                model = model_class(**hyperparams)
                                
                                start_time = time.time()
                                model.train(X_train, y_train)
                                training_time = time.time() - start_time
                            
                            # Evaluate
                            progress_bar.progress(85, "Evaluating...")
                            metrics = model.evaluate(X_test, y_test)
                            
                            progress_bar.progress(100, "Complete!")
                            time.sleep(0.5)
                            progress_bar.empty()
                            
                            # Store results
                            st.session_state['model'] = model
                            st.session_state['metrics'] = metrics
                            st.session_state['training_time'] = training_time
                            st.session_state['feature_names'] = df.columns.tolist()
                            if feature_selector is None:
                                st.session_state['selected_feature_names'] = None
                            st.session_state['trained'] = True
                            st.session_state['model_name'] = selected_model
                            st.session_state['best_params'] = best_params
                            st.session_state['cv_score'] = cv_score
                            
                            # Success banner
                            st.markdown(f"""
                                <div class="result-banner">
                                    <span class="result-banner-icon">✅</span>
                                    <span class="result-banner-text">
                                        Training complete! Achieved 
                                        <span class="result-banner-highlight">{metrics['accuracy']*100:.2f}%</span> 
                                        accuracy in 
                                        <span class="result-banner-highlight">{training_time:.4f}s</span>
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Show best params if auto-tuned
                            if best_params:
                                st.success(f"🎯 **Best Parameters Found (CV Score: {cv_score:.4f}):**")
                                params_str = ", ".join([f"`{k}={v}`" for k, v in best_params.items()])
                                st.markdown(params_str)
                            
                            st.info("💡 Switch to the **Results** tab to view detailed metrics and visualizations")
            
            # ---------- Compare All Models Table ----------
            st.markdown("---")
            st.markdown('<h2 class="section-title">📊 Compare All Models</h2>', unsafe_allow_html=True)
            st.markdown("Train all models with default parameters and see accuracy, precision, recall, F1, and ROC-AUC in one table.")
            compare_button = st.button("🔄 Run & Compare All Models", type="secondary", use_container_width=False)
            
            if compare_button and df.shape[1] > 0:
                with st.spinner("Training all models... This may take a minute."):
                    progress = st.progress(0)
                    y_vals = target.values
                    uniq, counts = np.unique(y_vals, return_counts=True)
                    can_stratify = len(uniq) > 1 and (counts >= 2).all()
                    X_train, X_test, y_train, y_test = train_test_split(
                        df.values, y_vals,
                        test_size=test_size,
                        random_state=random_state,
                        stratify=y_vals if can_stratify else None
                    )
                    use_smote = st.session_state.get('use_smote', False) and st.session_state.get('class_imbalance', False)
                    if use_smote and SMOTE_AVAILABLE:
                        try:
                            smote = SMOTE(random_state=random_state)
                            X_train, y_train = smote.fit_resample(X_train, y_train)
                        except Exception:
                            pass
                    n_samples, n_cols = X_train.shape[0], X_train.shape[1]
                    k_select = min(n_features, n_cols, max(5, n_samples // 50))
                    if use_feature_selection and n_cols > 5 and k_select < n_cols:
                        try:
                            fs = SelectKBest(score_func=mutual_info_classif, k=k_select)
                            X_train = fs.fit_transform(X_train, y_train)
                            X_test = fs.transform(X_test)
                        except Exception:
                            pass
                    if scale_method == "StandardScaler":
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    elif scale_method == "RobustScaler":
                        scaler = RobustScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    elif scale_method == "MinMaxScaler":
                        scaler = MinMaxScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    if X_train.shape[1] == 0:
                        st.error("No features remaining. Cannot compare models.")
                    else:
                        results = []
                        for idx, (model_name, model_class) in enumerate(MODELS.items()):
                            progress.progress((idx + 1) / len(MODELS), text=f"Training {model_name}...")
                            param_info = model_class.get_param_info()
                            hyperparams = {p: param_info[p]['default'] for p in param_info}
                            try:
                                model = model_class(**hyperparams)
                                t0 = time.time()
                                model.train(X_train, y_train)
                                train_time = time.time() - t0
                                metrics = model.evaluate(X_test, y_test)
                                results.append({
                                    'Model': model_name,
                                    'Accuracy': metrics['accuracy'],
                                    'Precision': metrics['precision'],
                                    'Recall': metrics['recall'],
                                    'F1 Score': metrics['f1_score'],
                                    'ROC-AUC': metrics.get('roc_auc') if metrics.get('roc_auc') is not None else np.nan,
                                    'Training Time (s)': round(train_time, 4)
                                })
                            except Exception as e:
                                results.append({
                                    'Model': model_name,
                                    'Accuracy': np.nan,
                                    'Precision': np.nan,
                                    'Recall': np.nan,
                                    'F1 Score': np.nan,
                                    'ROC-AUC': np.nan,
                                    'Training Time (s)': np.nan
                                })
                                st.warning(f"{model_name}: {str(e)}")
                        progress.empty()
                        
                        comparison_df = pd.DataFrame(results)
                        st.markdown("**All models comparison**")
                        st.dataframe(
                            comparison_df.style.format(
                                subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
                                formatter="{:.4f}"
                            ).format(subset=['Training Time (s)'], formatter="{:.4f}").highlight_max(
                                subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
                                axis=0,
                                color='rgba(124, 58, 237, 0.3)'
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                        best_acc_idx = comparison_df['Accuracy'].idxmax()
                        if pd.notna(comparison_df.loc[best_acc_idx, 'Accuracy']):
                            st.success(f"🏆 Best accuracy: **{comparison_df.loc[best_acc_idx, 'Model']}** ({comparison_df.loc[best_acc_idx, 'Accuracy']:.4f})")
            elif compare_button and df.shape[1] == 0:
                st.error("No features to train on. Upload a file with at least one feature column.")
        
        with tab3:
            if st.session_state.get('trained', False):
                metrics = st.session_state['metrics']
                training_time = st.session_state['training_time']
                model = st.session_state['model']
                feature_names = st.session_state['feature_names']
                model_name = st.session_state.get('model_name', 'Model')
                
                st.markdown(f'<h2 class="section-title">{MODEL_ICONS.get(model_name, "📊")} {model_name} Results</h2>', unsafe_allow_html=True)
                
                # Metric cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    render_metric_card("🎯", metrics['accuracy'], "Accuracy", "accuracy")
                with col2:
                    render_metric_card("✨", metrics['precision'], "Precision", "precision")
                with col3:
                    render_metric_card("🔄", metrics['recall'], "Recall", "recall")
                with col4:
                    render_metric_card("⚡", metrics['f1_score'], "F1 Score", "f1")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # ROC-AUC and training time
                if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
                    st.markdown(f"""
                        <div class="result-banner">
                            <span class="result-banner-icon">📊</span>
                            <span class="result-banner-text">
                                ROC-AUC: <span class="result-banner-highlight">{metrics['roc_auc']:.4f}</span>
                                &nbsp;•&nbsp;
                                Training Time: <span class="result-banner-highlight">{training_time:.4f}s</span>
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_cm = create_confusion_matrix_plot(metrics['confusion_matrix'])
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with col2:
                    fig_radar = create_metrics_radar_plot(metrics)
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # Feature importance (use selected feature names if feature selection was used)
                importance_vals = model.get_feature_importance() if hasattr(model, 'get_feature_importance') else None
                if importance_vals is not None:
                    names_for_importance = st.session_state.get('selected_feature_names') or feature_names
                    if len(importance_vals) != len(names_for_importance):
                        names_for_importance = names_for_importance[:len(importance_vals)]
                    st.markdown('<h2 class="section-title">Feature Importance</h2>', unsafe_allow_html=True)
                    fig_importance = create_feature_importance_plot(importance_vals, names_for_importance)
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # Summary table
                st.markdown('<h2 class="section-title">Performance Summary</h2>', unsafe_allow_html=True)
                
                summary_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time'],
                    'Value': [
                        f"{metrics['accuracy']:.4f}",
                        f"{metrics['precision']:.4f}",
                        f"{metrics['recall']:.4f}",
                        f"{metrics['f1_score']:.4f}",
                        f"{training_time:.4f}s"
                    ],
                    'Rating': [
                        '🟢 Excellent' if metrics['accuracy'] > 0.9 else '🟡 Good' if metrics['accuracy'] > 0.7 else '🔴 Needs Improvement',
                        '🟢 Excellent' if metrics['precision'] > 0.9 else '🟡 Good' if metrics['precision'] > 0.7 else '🔴 Needs Improvement',
                        '🟢 Excellent' if metrics['recall'] > 0.9 else '🟡 Good' if metrics['recall'] > 0.7 else '🔴 Needs Improvement',
                        '🟢 Excellent' if metrics['f1_score'] > 0.9 else '🟡 Good' if metrics['f1_score'] > 0.7 else '🔴 Needs Improvement',
                        '🟢 Fast' if training_time < 1 else '🟡 Moderate' if training_time < 5 else '🔴 Slow'
                    ]
                }
                
                if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
                    summary_data['Metric'].insert(4, 'ROC-AUC')
                    summary_data['Value'].insert(4, f"{metrics['roc_auc']:.4f}")
                    summary_data['Rating'].insert(4, '🟢 Excellent' if metrics['roc_auc'] > 0.9 else '🟡 Good' if metrics['roc_auc'] > 0.7 else '🔴 Needs Improvement')
                
                st.dataframe(
                    pd.DataFrame(summary_data),
                    use_container_width=True,
                    hide_index=True
                )
            
            else:
                st.markdown("""
                    <div class="empty-state">
                        <span class="empty-state-icon">🔬</span>
                        <h3 class="empty-state-title">No Results Yet</h3>
                        <p class="empty-state-desc">
                            Train a model in the "Train Model" tab to see detailed performance metrics and visualizations here.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    
    else:
        # Empty state - no data loaded
        st.markdown("""
            <div class="empty-state">
                <span class="empty-state-icon">📁</span>
                <h3 class="empty-state-title">Get Started</h3>
                <p class="empty-state-desc">
                    Upload your CSV file from the sidebar to begin exploring machine learning models.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <p style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.8rem; color: #71717a;">
                Built with Streamlit • ML Classification Playground
            </p>
            <p style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.75rem; color: #71717a; margin-top: 0.5rem;">
                <a href="https://github.com/2025aa05482-bits/ML-assignment2" target="_blank" rel="noopener noreferrer" style="color: #7c3aed; text-decoration: none;">📂 GitHub Repository</a>
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
