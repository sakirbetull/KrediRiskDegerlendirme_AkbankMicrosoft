import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
from config.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET

def load_data(file_path):
    """Load the dataset from the given file path."""
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Fill missing values in loan_int_rate with median
    df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
    # Fill missing values in person_emp_length with median
    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
    return df

def handle_outliers(df):
    """Handle outliers in numerical features using winsorization."""
    for feature in NUMERICAL_FEATURES:
        df[feature] = winsorize(df[feature], limits=[0.05, 0.05])
    return df

def preprocess_data(df):
    """Main preprocessing function that combines all preprocessing steps."""
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def prepare_features(df):
    """Prepare features for model training."""
    # One-hot encoding for categorical features
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
    
    # Standard scaling for numerical features
    scaler = StandardScaler()
    df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])
    
    return df, scaler 