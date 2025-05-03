import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def plot_numerical_distributions(df):
    """Plot distributions of numerical features."""
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(NUMERICAL_FEATURES, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df):
    """Plot distributions of categorical features."""
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(CATEGORICAL_FEATURES, 1):
        plt.subplot(2, 2, i)
        sns.countplot(data=df, x=feature)
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """Plot correlation matrix for numerical features."""
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[NUMERICAL_FEATURES].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_target_distribution(df, target_column):
    """Plot distribution of target variable."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=target_column)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Target Value')
    plt.ylabel('Count')
    plt.show()

def get_data_summary(df):
    """Get summary statistics of the dataset."""
    summary = {
        'shape': df.shape,
        'missing_values': df.isnull().sum(),
        'data_types': df.dtypes,
        'numerical_stats': df[NUMERICAL_FEATURES].describe(),
        'categorical_counts': {col: df[col].value_counts() for col in CATEGORICAL_FEATURES}
    }
    return summary 