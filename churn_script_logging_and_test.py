import os
import logging
from pathlib import Path
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering,\
	classification_report_image, feature_importance_plot, train_models
from pandas.api.types import is_numeric_dtype
DATA_PATH  = r"data\bank_data.csv"
BASE_PATH = r"C:/Work/Courses/Machine Learning DevOps Engineer/Predict customer churn/"
EDA_PATH = "images\eda"
RESULTS_PATH = "images\results"
MODELS_PATH = "models"

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data(DATA_PATH)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err

def test_eda():
    df = import_data(DATA_PATH)

    perform_eda(df)

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

def test_encoder_helper():
    df = import_data(DATA_PATH)

    category_lst = df.select_dtypes(["object"]).columns
    cat_lst = [cat +'_Churn' for cat in category_lst]

    df = encoder_helper(df)
    cat_encoded_count = 0

    for column in df.columns:
        if column in cat_lst:
            # assert if is a numeric value
            assert is_numeric_dtype(df[column])
            cat_encoded_count +=1

    assert cat_encoded_count == len(category_lst)

def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df = import_data(DATA_PATH)
    df = encoder_helper(df)

    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    
    try:
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
    except AssertionError as err:
        logging.error("One or more sets are empty")
        raise err



def test_train_models():
    '''
    test train_models
    '''
    df = import_data(DATA_PATH)
    df = encoder_helper(df)

    X_train, X_test, y_train, y_test = perform_feature_engineering(df)

    train_models(X_train, X_test, y_train, y_test)

    feature_importance_plot = Path(RESULTS_PATH, "feature_importance.png")
    roc_plot = Path(RESULTS_PATH, "roc_plot.png")

    random_forest_model = Path(MODELS_PATH, "rfc_model.pkl")
    logistic_regression_model = Path(MODELS_PATH, "logistic_model.pkl")
    
    assert feature_importance_plot.exists
    assert roc_plot.exists
    assert random_forest_model.exists
    assert logistic_regression_model.exists


def test_classification_report_images():
    '''
    test classification_report_images
    '''
    df = import_data(DATA_PATH)
    df = encoder_helper(df)

    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    

    random_forest_report = Path(EDA_PATH, "rf_class_rpt.png")
    logistic_regression_report = Path(EDA_PATH, "lr_class_rpt.png")

    assert random_forest_report.exists()
    assert logistic_regression_report.exists()
if __name__ == "__main__":
    pass