"""Hard-coded results from the LAMBDA paper (arxiv 2407.17535) Table 2-9 for comparison."""

# Table 2: Classification (Accuracy %)
PAPER_CLASSIFICATION = {
    "AIDS Clinical Trials Group Study 175": {
        "Logistic Regression": 86.54, "SVM": 88.45, "Neural Network (MLPClassifier)": 88.82,
        "Decision Tree": 87.70, "Random Forest": 89.29, "Bagging": 89.62,
        "Gradient Boosting": 89.20, "XGBoost": 89.67, "AdaBoost": 88.92,
    },
    "NHANES Age Prediction": {
        "Logistic Regression": 99.43, "SVM": 98.82, "Neural Network (MLPClassifier)": 99.91,
        "Decision Tree": 100, "Random Forest": 100, "Bagging": 100,
        "Gradient Boosting": 100, "XGBoost": 100, "AdaBoost": 100,
    },
    "Breast Cancer Wisconsin Diagnostic": {
        "Logistic Regression": 98.07, "SVM": 97.72, "Neural Network (MLPClassifier)": 97.82,
        "Decision Tree": 94.26, "Random Forest": 96.84, "Bagging": 96.49,
        "Gradient Boosting": 96.84, "XGBoost": 97.54, "AdaBoost": 97.72,
    },
    "Wine": {
        "Logistic Regression": 98.89, "SVM": 98.89, "Neural Network (MLPClassifier)": 82.60,
        "Decision Tree": 92.14, "Random Forest": 98.33, "Bagging": 96.65,
        "Gradient Boosting": 96.65, "XGBoost": 95.54, "AdaBoost": 93.89,
    },
}

# Table 2: Regression (MSE)
PAPER_REGRESSION = {
    "Concrete Compressive Strength": {
        "Linear Regression": 0.4596, "Lasso": 0.5609, "SVR": 0.4012,
        "Neural Network (MLPRegressor)": 0.2749, "Decision Tree": 0.5242,
        "Random Forest": 0.4211, "Gradient Boosting": 0.3414,
        "XGBoost": 0.3221, "CatBoost": 0.2876,
    },
    "Combined Cycle Power Plant": {
        "Linear Regression": 0.0714, "Lasso": 0.0718, "SVR": 0.0534,
        "Neural Network (MLPRegressor)": 0.0612, "Decision Tree": 0.0551,
        "Random Forest": 0.0375, "Gradient Boosting": 0.0315,
        "XGBoost": 0.0319, "CatBoost": 0.0325,
    },
    "Abalone": {
        "Linear Regression": 0.5086, "Lasso": 0.8042, "SVR": 0.4542,
        "Neural Network (MLPRegressor)": 0.4551, "Decision Tree": 0.5566,
        "Random Forest": 0.4749, "Gradient Boosting": 0.4778,
        "XGBoost": 0.4778, "CatBoost": 0.4795,
    },
    "Airfoil Self-Noise": {
        "Linear Regression": 0.5717, "Lasso": 0.5738, "SVR": 0.3854,
        "Neural Network (MLPRegressor)": 0.4292, "Decision Tree": 0.3823,
        "Random Forest": 0.2655, "Gradient Boosting": 0.2528,
        "XGBoost": 0.2741, "CatBoost": 0.2529,
    },
}

# Table 4: High-Dimensional Genomic (Accuracy %)
PAPER_GENOMIC = {
    "TCGAmirna": {
        "Logistic Regression": 52.58, "SVM": 54.42, "Neural Network (MLPClassifier)": 54.22,
        "Decision Tree": 54.42, "Random Forest": 55.16, "Bagging": 56.62,
        "Gradient Boosting": 54.78, "XGBoost": 55.15, "AdaBoost": 55.15,
    },
    "EMTAB386": {
        "Logistic Regression": 54.18, "SVM": 57.45, "Neural Network (MLPClassifier)": 61.23,
        "Decision Tree": 57.45, "Random Forest": 61.20, "Bagging": 58.21,
        "Gradient Boosting": 55.08, "XGBoost": 58.15, "AdaBoost": 57.45,
    },
    "GSE49997": {
        "Logistic Regression": 67.52, "SVM": 63.45, "Neural Network (MLPClassifier)": 66.48,
        "Decision Tree": 63.45, "Random Forest": 67.54, "Bagging": 70.63,
        "Gradient Boosting": 70.62, "XGBoost": 70.62, "AdaBoost": 70.62,
    },
}

# Table 5: Missing Data (Accuracy %)
PAPER_MISSING = {
    "Framingham Heart Study": {
        "Logistic Regression": 85.35, "SVM": 84.95, "Neural Network (MLPClassifier)": 84.95,
        "Decision Tree": 84.27, "Random Forest": 85.19, "Bagging": 85.02,
        "Gradient Boosting": 85.12, "XGBoost": 85.19, "AdaBoost": 84.98,
    },
    "Student Admission Records": {
        "Logistic Regression": 50.36, "SVM": 57.28, "Neural Network (MLPClassifier)": 57.28,
        "Decision Tree": 52.96, "Random Forest": 55.40, "Bagging": 58.65,
        "Gradient Boosting": 60.50, "XGBoost": 61.05, "AdaBoost": 56.63,
    },
    "Heart Disease": {
        "Logistic Regression": 59.41, "SVM": 60.40, "Neural Network (MLPClassifier)": 60.40,
        "Decision Tree": 52.49, "Random Forest": 60.39, "Bagging": 60.06,
        "Gradient Boosting": 58.41, "XGBoost": 60.71, "AdaBoost": 59.42,
    },
}

# Table 6: MNIST (Accuracy %)
PAPER_IMAGE = {
    "MNIST CNN": {"CNN": 99.19},
    "MNIST Transformer": {"Transformer": 97.23},
}

# Table 7: SMS Spam (Accuracy %)
PAPER_TEXT = {
    "SMS Spam Naive Bayes": {"Multinomial Naive Bayes": 98.39},
    "SMS Spam BERT": {"DistilBERT": 99.37},
}

# Table 8: Knowledge Integration (Score 0-1)
PAPER_KNOWLEDGE = {
    "Pattern Mining (PAMI)": {"PAMI FPGrowth": 1.00},
    "Nearest Correlation Matrix": {"Newton Method": 1.00},
    "Non-negative Neural Networks": {"NonNeg NN": 1.00},
}


def get_paper_result(dataset_name: str, model_name: str) -> float:
    """Look up paper result for a given dataset+model combination."""
    for results_dict in [PAPER_CLASSIFICATION, PAPER_REGRESSION, PAPER_GENOMIC,
                         PAPER_MISSING, PAPER_IMAGE, PAPER_TEXT, PAPER_KNOWLEDGE]:
        if dataset_name in results_dict:
            models = results_dict[dataset_name]
            if model_name in models:
                return models[model_name]
    return None
