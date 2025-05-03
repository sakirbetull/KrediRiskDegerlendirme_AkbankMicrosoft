# Credit Risk Assessment Project

This project implements a machine learning solution for credit risk assessment using various classification algorithms. The goal is to predict whether a loan applicant is likely to default on their loan.

## Project Structure

```
credit_risk_assessment/
├── data/                    # Data directory
│   └── credit_risk_dataset.csv
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── utils.py
├── notebooks/               # Jupyter notebooks
│   └── exploratory_data_analysis.ipynb
├── models/                  # Trained models
│   └── saved_models/
├── config/                  # Configuration files
│   └── config.py
├── tests/                   # Test files
│   └── test_models.py
├── requirements.txt         # Project dependencies
└── main.py                 # Main script
```

## Features

- Data preprocessing and cleaning
- Exploratory data analysis
- Feature engineering
- Multiple model training and evaluation
- Model performance visualization
- Model persistence

## Dataset

The dataset contains the following features:

- `person_age`: Age of the person
- `person_income`: Annual income
- `person_home_ownership`: Home ownership status
- `person_emp_length`: Employment length in years
- `loan_intent`: Purpose of the loan
- `loan_grade`: Loan grade
- `loan_amnt`: Loan amount
- `loan_int_rate`: Interest rate
- `loan_status`: Target variable (0: No default, 1: Default)
- `loan_percent_income`: Loan amount as percentage of income
- `cb_person_default_on_file`: Historical default
- `cb_person_cred_hist_length`: Credit history length

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-risk-assessment.git
cd credit-risk-assessment
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python main.py
```

This will:
- Load and preprocess the data
- Perform exploratory data analysis
- Train multiple models
- Evaluate model performance
- Save trained models

## Models

The project implements the following models:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

## Evaluation Metrics

Models are evaluated using:
- Classification Report
- Confusion Matrix
- ROC AUC Score
- Feature Importance

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- Thanks to all contributors who have helped with this project 