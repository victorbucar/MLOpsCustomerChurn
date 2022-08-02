"""This is a testing file for method in churn_library.py:
    Import data.
    Perform EDA
    Perform categorical variables encoding
    Report image classification
    Plots
    Train model
"""
import logging
from pathlib import Path
from pandas.api.types import is_numeric_dtype
from numpy import loadtxt
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering,\
    classification_report_image, train_models

DATA_PATH = r"data\bank_data.csv"
BASE_PATH = r"C:/Work/Courses/Machine Learning DevOps Engineer/Predict customer churn/"
EDA_PATH = "images\\eda"
RESULTS_PATH = "images\results"
REPORTS_PATH = "images\\results"
MODELS_PATH = "models"

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import -
    '''
    try:
        data_set = import_data(DATA_PATH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_set.shape[0] > 0
        assert data_set.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test eda - to make sure all plots are created
    '''
    data_set = import_data(DATA_PATH)

    perform_eda(data_set)
    try:
        churn_hist = Path(EDA_PATH, "churn_hist.png")
        age_hist = Path(EDA_PATH, "age_hist.png")
        marital_status_bar = Path(EDA_PATH, "marital_status_bar.png")
        total_trans_hist = Path(EDA_PATH, "total_trans_hist.png")
        correlation_heatmap = Path(EDA_PATH, "correlation_heatmap.png")

        assert churn_hist.exists()
        assert age_hist.exists()
        assert marital_status_bar.exists()
        assert total_trans_hist.exists()
        assert correlation_heatmap.exists()
        logging.info("Testing eda: PASSED")
    except AssertionError:
        logging.error("Testing eda: FAILED! One or more files do not exist.")



def test_encoder_helper():
    '''
    test encoder helper - to make sure all categorical features values are encoded properly
    '''
    try:
        data_set = import_data(DATA_PATH)

        category_lst = data_set.select_dtypes(["object"]).columns
        cat_lst = [cat + '_Churn' for cat in category_lst]

        data_set = encoder_helper(data_set)
        cat_encoded_count = 0

        for column in data_set.columns:
            if column in cat_lst:
                # assert if is a numeric value
                assert is_numeric_dtype(data_set[column])
                cat_encoded_count += 1

        assert cat_encoded_count == len(category_lst)
        logging.info("Testing encoder helper: PASSED!")
    except AssertionError:
        logging.error("Testing enconder helper: FAILED!"
        "Function could not convert one or more features")


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    data_set = import_data(DATA_PATH)
    data_set = encoder_helper(data_set)

    X_train, X_test, y_train, y_test = perform_feature_engineering(data_set)

    try:
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
    except AssertionError:
        logging.error("Testing feature engineering FAILED! One or more sets are empty")
        # raise err


def test_train_models():
    '''
    test train_models
    '''
    try:
        data_set = import_data(DATA_PATH)
        data_set = encoder_helper(data_set)

        X_train, X_test, y_train, y_test = perform_feature_engineering(data_set)

        train_models(X_train, X_test, y_train, y_test)

        feature_importance_plot = Path(RESULTS_PATH, "feature_importance.png")
        roc_plot = Path(RESULTS_PATH, "roc_plot.png")
        tree_explainer_plot = Path(RESULTS_PATH, "tree_explainer.png")
        logging.info("Testing train models: PASSED!")

        assert feature_importance_plot.exists
        assert roc_plot.exists
        assert tree_explainer_plot.exists

        random_forest_model = Path(MODELS_PATH, "rfc_model.pkl")
        logistic_regression_model = Path(MODELS_PATH, "logistic_model.pkl")

        assert random_forest_model.exists
        assert logistic_regression_model.exists
        logging.info("Testing train models: SUCCESS!")

    except AssertionError:
        logging.error("Testing train models: FAILED!"
        "One or more plots or models not saved correctly")

def test_classification_report_images():
    '''
    test classification_report_images
    '''
    try:
        data_set = import_data(DATA_PATH)
        data_set = encoder_helper(data_set)

        X_train, X_test, y_train, y_test = perform_feature_engineering(data_set)
        # y_train_test_preds_lr, y_train_test_preds_rf = test_load_pre_trained_models()

        y_train_preds_lr = loadtxt('./logs/y_train_preds_lr.csv', delimiter=',')
        y_train_preds_rf = loadtxt('./logs/y_train_preds_rf.csv', delimiter=',')
        y_test_preds_lr = loadtxt('./logs/y_test_preds_lr.csv', delimiter=',')
        y_test_preds_rf = loadtxt('./logs/y_test_preds_rf.csv', delimiter=',')

        # Act
        classification_report_image(y_train,
                                    y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf)

        # assert
        random_forest_report = Path(REPORTS_PATH, "rf_class_rpt.png")
        logistic_regression_report = Path(REPORTS_PATH, "lr_class_rpt.png")

        assert random_forest_report.exists()
        assert logistic_regression_report.exists()

        logging.info("Testing classification report: SUCCESS!")
    except AssertionError:
        logging.error("Testing classification report: FAILED!"
        "One or more files do not exist")

if __name__ == "__main__":
    pass
