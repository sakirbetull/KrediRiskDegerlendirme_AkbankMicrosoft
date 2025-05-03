import pandas as pd
from src.data_preprocessing import load_data, preprocess_data, prepare_features
from src.model_training import create_balanced_dataset, split_data, train_models, evaluate_models, save_models
from src.evaluation import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, print_model_metrics
from src.utils import plot_numerical_distributions, plot_categorical_distributions, plot_correlation_matrix, plot_target_distribution, get_data_summary
from config.config import DATA_FILE, TARGET

def main():
    # Load data
    print("Loading data...")
    df = load_data(DATA_FILE)
    
    # Exploratory Data Analysis
    print("\nPerforming Exploratory Data Analysis...")
    plot_numerical_distributions(df)
    plot_categorical_distributions(df)
    plot_correlation_matrix(df)
    plot_target_distribution(df, TARGET)
    
    # Print data summary
    summary = get_data_summary(df)
    print("\nData Summary:")
    print(f"Dataset Shape: {summary['shape']}")
    print("\nMissing Values:")
    print(summary['missing_values'])
    print("\nData Types:")
    print(summary['data_types'])
    
    # Preprocess data
    print("\nPreprocessing data...")
    df = preprocess_data(df)
    
    # Prepare features
    print("\nPreparing features...")
    df, scaler = prepare_features(df)
    
    # Create balanced dataset
    print("\nCreating balanced dataset...")
    balanced_df = create_balanced_dataset(df)
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(balanced_df)
    
    # Train models
    print("\nTraining models...")
    models = train_models(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_test)
    
    # Print model metrics
    print_model_metrics(results)
    
    # Plot evaluation metrics
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        plot_confusion_matrix(results[model_name]['confusion_matrix'], model_name)
        plot_roc_curve(y_test, y_pred_proba, model_name)
        plot_feature_importance(model, X_train.columns, model_name)
    
    # Save models
    print("\nSaving models...")
    save_models(models)
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()
