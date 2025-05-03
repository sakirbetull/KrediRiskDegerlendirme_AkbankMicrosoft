import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from config.config import RANDOM_STATE, TEST_SIZE, MODELS_DIR, TARGET

def create_balanced_dataset(df, target_column=TARGET):
    """Create a balanced dataset by sampling equal number of instances from each class."""
    class_0 = df[df[target_column] == 0]
    class_1 = df[df[target_column] == 1]
    
    # Sample equal number of instances from each class
    min_samples = min(len(class_0), len(class_1))
    class_0_sample = class_0.sample(n=min_samples, random_state=RANDOM_STATE)
    class_1_sample = class_1.sample(n=min_samples, random_state=RANDOM_STATE)
    
    # Combine and shuffle the samples
    balanced_data = pd.concat([class_0_sample, class_1_sample])
    balanced_data = balanced_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    return balanced_data

def split_data(df, target_column=TARGET):
    """Split the data into training and testing sets."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def train_models(X_train, y_train):
    """Train multiple models and return them."""
    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and return their performance metrics."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc_score': roc_auc_score(y_test, y_pred_proba)
        }
    
    return results

def save_models(models):
    """Save trained models to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f'{name.lower().replace(" ", "_")}.joblib')) 