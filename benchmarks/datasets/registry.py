# Benchmark dataset metadata registry
# Each entry: name, source, url, filename, target_col, task_type, metric, group, models

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DatasetInfo:
    name: str
    group: str           # group1_classical, group2_regression, group3_genomic, etc.
    task_type: str       # classification or regression
    metric: str          # accuracy or mse
    source: str          # uci, kaggle, torch, synthetic
    url: str
    filename: str
    target_col: str
    models: List[str]    # which ML models to test
    extra_instruction: str = ""  # additional context for the LLM
    uci_id: Optional[int] = None

CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "SVM",
    "Neural Network (MLPClassifier)",
    "Decision Tree",
    "Random Forest",
    "Bagging",
    "Gradient Boosting",
    "XGBoost",
    "AdaBoost",
]

REGRESSION_MODELS = [
    "Linear Regression",
    "Lasso",
    "SVR",
    "Neural Network (MLPRegressor)",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "XGBoost Regressor",
    "CatBoost",
]

# Map model name → sklearn/xgboost class hint for instruction
MODEL_IMPORTS = {
    "Logistic Regression": "from sklearn.linear_model import LogisticRegression",
    "SVM": "from sklearn.svm import SVC",
    "Neural Network (MLPClassifier)": "from sklearn.neural_network import MLPClassifier",
    "Decision Tree": "from sklearn.tree import DecisionTreeClassifier",
    "Random Forest": "from sklearn.ensemble import RandomForestClassifier",
    "Bagging": "from sklearn.ensemble import BaggingClassifier",
    "Gradient Boosting": "from sklearn.ensemble import GradientBoostingClassifier",
    "XGBoost": "from xgboost import XGBClassifier",
    "AdaBoost": "from sklearn.ensemble import AdaBoostClassifier",
    "Linear Regression": "from sklearn.linear_model import LinearRegression",
    "Lasso": "from sklearn.linear_model import Lasso",
    "SVR": "from sklearn.svm import SVR",
    "Neural Network (MLPRegressor)": "from sklearn.neural_network import MLPRegressor",
    "Decision Tree Regressor": "from sklearn.tree import DecisionTreeRegressor",
    "Random Forest Regressor": "from sklearn.ensemble import RandomForestRegressor",
    "Gradient Boosting Regressor": "from sklearn.ensemble import GradientBoostingRegressor",
    "XGBoost Regressor": "from xgboost import XGBRegressor",
    "CatBoost": "from catboost import CatBoostRegressor",
}

ALL_DATASETS = [
    # ========================= GROUP 1: Classical Tabular Classification =========================
    DatasetInfo(
        name="AIDS Clinical Trials Group Study 175",
        group="group1_classification",
        task_type="classification",
        metric="accuracy",
        source="uci",
        url="https://archive.ics.uci.edu/static/markup/890",
        filename="aids_clinical.csv",
        target_col="cid",
        models=CLASSIFICATION_MODELS,
        uci_id=890,
    ),
    DatasetInfo(
        name="NHANES Age Prediction",
        group="group1_classification",
        task_type="classification",
        metric="accuracy",
        source="uci",
        url="https://archive.ics.uci.edu/static/markup/887",
        filename="nhanes.csv",
        target_col="age_group",
        models=CLASSIFICATION_MODELS,
        uci_id=887,
    ),
    DatasetInfo(
        name="Breast Cancer Wisconsin Diagnostic",
        group="group1_classification",
        task_type="classification",
        metric="accuracy",
        source="uci",
        url="https://archive.ics.uci.edu/static/markup/17",
        filename="breast_cancer.csv",
        target_col="Diagnosis",
        models=CLASSIFICATION_MODELS,
        uci_id=17,
    ),
    DatasetInfo(
        name="Wine",
        group="group1_classification",
        task_type="classification",
        metric="accuracy",
        source="uci",
        url="https://archive.ics.uci.edu/static/markup/109",
        filename="wine.csv",
        target_col="class",
        models=CLASSIFICATION_MODELS,
        uci_id=109,
    ),

    # ========================= GROUP 2: Classical Tabular Regression =========================
    DatasetInfo(
        name="Concrete Compressive Strength",
        group="group2_regression",
        task_type="regression",
        metric="mse",
        source="uci",
        url="https://archive.ics.uci.edu/static/markup/165",
        filename="concrete.csv",
        target_col="Concrete compressive strength",
        models=REGRESSION_MODELS,
        extra_instruction="The target column name has spaces: 'Concrete compressive strength'. Use it exactly.",
        uci_id=165,
    ),
    DatasetInfo(
        name="Combined Cycle Power Plant",
        group="group2_regression",
        task_type="regression",
        metric="mse",
        source="uci",
        url="https://archive.ics.uci.edu/static/markup/294",
        filename="power_plant.csv",
        target_col="PE",
        models=REGRESSION_MODELS,
        uci_id=294,
    ),
    DatasetInfo(
        name="Abalone",
        group="group2_regression",
        task_type="regression",
        metric="mse",
        source="uci",
        url="https://archive.ics.uci.edu/static/markup/1",
        filename="abalone.csv",
        target_col="Rings",
        models=REGRESSION_MODELS,
        extra_instruction="Encode the 'Sex' column using one-hot encoding before training.",
        uci_id=1,
    ),
    DatasetInfo(
        name="Airfoil Self-Noise",
        group="group2_regression",
        task_type="regression",
        metric="mse",
        source="uci",
        url="https://archive.ics.uci.edu/static/markup/291",
        filename="airfoil.csv",
        target_col="scaled-sound-pressure",
        models=REGRESSION_MODELS,
        extra_instruction="The target column is 'scaled-sound-pressure'.",
        uci_id=291,
    ),

    # ========================= GROUP 3: High-Dimensional Genomic =========================
    DatasetInfo(
        name="TCGAmirna",
        group="group3_genomic",
        task_type="classification",
        metric="accuracy",
        source="kaggle",
        url="https://www.kaggle.com/datasets/anhpknu/high-dimensional-data",
        filename="tcga_mirna.csv",
        target_col="vital_status",
        models=CLASSIFICATION_MODELS,
        extra_instruction="This is a high-dimensional genomic dataset (544 rows x 802 features). The target column 'vital_status' should be encoded as binary. Apply PCA dimensionality reduction before classification. Drop the 'Unnamed: 0' column first.",
    ),
    DatasetInfo(
        name="EMTAB386",
        group="group3_genomic",
        task_type="classification",
        metric="accuracy",
        source="kaggle",
        url="https://www.kaggle.com/datasets/anhpknu/high-dimensional-data",
        filename="emtab386.csv",
        target_col="event",
        models=CLASSIFICATION_MODELS,
        extra_instruction="This is a high-dimensional genomic dataset (129 rows x 10360 features). The target column is 'event'. Apply PCA dimensionality reduction before classification. Drop the 'Unnamed: 0' column first.",
    ),
    DatasetInfo(
        name="GSE49997",
        group="group3_genomic",
        task_type="classification",
        metric="accuracy",
        source="kaggle",
        url="https://www.kaggle.com/datasets/anhpknu/high-dimensional-data",
        filename="gse49997.csv",
        target_col="event",
        models=CLASSIFICATION_MODELS,
        extra_instruction="This is a high-dimensional genomic dataset (194 rows x 16051 features). The target column is 'event'. Apply PCA dimensionality reduction before classification. Drop the 'Unnamed: 0' column first.",
    ),

    # ========================= GROUP 4: Missing Data =========================
    DatasetInfo(
        name="Framingham Heart Study",
        group="group4_missing",
        task_type="classification",
        metric="accuracy",
        source="kaggle",
        url="https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset",
        filename="framingham.csv",
        target_col="TenYearCHD",
        models=CLASSIFICATION_MODELS,
        extra_instruction="This dataset contains missing values. Handle them by imputing with mean value or dropping rows with missing values before training.",
    ),
    DatasetInfo(
        name="Student Admission Records",
        group="group4_missing",
        task_type="classification",
        metric="accuracy",
        source="kaggle",
        url="https://www.kaggle.com/datasets/mohansacharya/graduate-admissions",
        filename="student_admission.csv",
        target_col="Chance of Admit ",
        models=CLASSIFICATION_MODELS,
        extra_instruction="The target column is 'Chance of Admit ' (note the trailing space). Convert it to binary classification (>=0.5 = admitted, <0.5 = not admitted). Drop 'Serial No.' column. Handle any missing values.",
    ),
    DatasetInfo(
        name="Heart Disease",
        group="group4_missing",
        task_type="classification",
        metric="accuracy",
        source="uci",
        url="https://archive.ics.uci.edu/static/markup/45",
        filename="heart_disease.csv",
        target_col="num",
        models=CLASSIFICATION_MODELS,
        extra_instruction="This dataset contains missing values (marked as '?'). Handle them appropriately. The target column 'num' should be converted to binary (0 = no disease, 1-4 = disease).",
        uci_id=45,
    ),

    # ========================= GROUP 5: Image Data (MNIST) =========================
    DatasetInfo(
        name="MNIST CNN",
        group="group5_image",
        task_type="classification",
        metric="accuracy",
        source="torch",
        url="",
        filename="mnist_cnn",
        target_col="digit",
        models=["CNN"],
        extra_instruction="Load MNIST using torchvision.datasets. Train a Convolutional Neural Network (CNN). Use standard train/test split.",
    ),
    DatasetInfo(
        name="MNIST Transformer",
        group="group5_image",
        task_type="classification",
        metric="accuracy",
        source="torch",
        url="",
        filename="mnist_transformer",
        target_col="digit",
        models=["Transformer"],
        extra_instruction="Load MNIST using torchvision.datasets. Train a Transformer-based classifier. Use standard train/test split.",
    ),

    # ========================= GROUP 6: Text Data =========================
    DatasetInfo(
        name="SMS Spam Naive Bayes",
        group="group6_text",
        task_type="classification",
        metric="accuracy",
        source="uci",
        url="https://archive.ics.uci.edu/static/markup/228",
        filename="sms_spam.csv",
        target_col="label",
        models=["Multinomial Naive Bayes"],
        extra_instruction="Load the SMS Spam Collection dataset. Train a Multinomial Naive Bayes classifier. Use 5-fold CV.",
        uci_id=228,
    ),
    DatasetInfo(
        name="SMS Spam BERT",
        group="group6_text",
        task_type="classification",
        metric="accuracy",
        source="uci",
        url="https://archive.ics.uci.edu/static/markup/228",
        filename="sms_spam_bert",
        target_col="label",
        models=["DistilBERT"],
        extra_instruction="Load the SMS Spam Collection dataset. Use DistilBERT-base-uncased for transfer learning to classify spam. Use standard train/test split.",
        uci_id=228,
    ),

    # ========================= GROUP 7: Knowledge Integration =========================
    DatasetInfo(
        name="Pattern Mining (PAMI)",
        group="group7_knowledge",
        task_type="knowledge_integration",
        metric="score",
        source="synthetic",
        url="",
        filename="pattern_mining",
        target_col="",
        models=["PAMI FPGrowth"],
        extra_instruction="Use the PAMI library to perform pattern mining on a transactional database. Use FPGrowth with minSup=500.",
    ),
    DatasetInfo(
        name="Nearest Correlation Matrix",
        group="group7_knowledge",
        task_type="knowledge_integration",
        metric="score",
        source="synthetic",
        url="",
        filename="ncm",
        target_col="",
        models=["Newton Method"],
        extra_instruction="Compute the nearest correlation matrix using the Quadratically Convergent Newton Method. Use a 2000x2000 random symmetric matrix.",
    ),
    DatasetInfo(
        name="Non-negative Neural Networks",
        group="group7_knowledge",
        task_type="knowledge_integration",
        metric="score",
        source="synthetic",
        url="",
        filename="nn_network",
        target_col="",
        models=["NonNeg NN"],
        extra_instruction="Train a non-negative neural network that maps nonnegative vectors to nonnegative vectors.",
    ),
]


def get_datasets_by_group(group: str) -> List[DatasetInfo]:
    return [d for d in ALL_DATASETS if d.group == group]


def get_all_groups() -> List[str]:
    return sorted(set(d.group for d in ALL_DATASETS))
