"""This library serves as a helper to:
    Import data.
    Perform EDA
    Perform categorical variables encoding
    Report image classification
    Plots
    Train model
"""


# import libraries
import os
import logging
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from pathlib import Path

import pandas as pd
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

EDA_PATH = "images\eda"
REPORTS_PATH = "images\results"

os.environ['QT_QPA_PLATFORM']='offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

X = pd.DataFrame()

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        logging.info("Reading file from path: %s", pth)
        assert isinstance(pth, str)
        data = pd.read_csv(pth)
        data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
        return data
    except FileNotFoundError:
        logging.error("Could not load dataframe. File/Path not found!")

def perform_eda(pd_df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    fig = plt.figure(figsize=(20,10))
    pd_df['Churn'].hist()
    plt.savefig(Path(EDA_PATH, "churn_hist.png"))


    fig1 = plt.figure(figsize=(20,10))
    pd_df['Customer_Age'].hist()
    plt.savefig(Path(EDA_PATH, "age_hist.png"))


    fig2 = plt.figure(figsize=(20,10))
    pd_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    fig2.savefig(Path(EDA_PATH, "marital_status_bar.png"))


    fig3 = plt.figure(figsize=(20,10))
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using
    # a kernel density estimate
    sns.histplot(pd_df['Total_Trans_Ct'], stat='density', kde=True)
    fig3.savefig(Path(EDA_PATH, "total_trans_hist.png"))


    fig4 = plt.figure(figsize=(20,10))
    sns.heatmap(pd_df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    fig4.savefig(Path(EDA_PATH, "correlation_heatmap.png"))


def encoder_helper(data_frame):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    new_df = data_frame.copy()
    category_lst = data_frame.select_dtypes(["object"]).columns
    try:
        for category in category_lst:
            cat_lst = []
            cat_group = new_df.groupby(category).mean()['Churn']
            for val in new_df[category]:
                cat_lst.append(cat_group.loc[val])
            new_df[category+'_Churn'] = cat_lst
            logging.info("Created new column %s", category+'_Churn')
    except KeyError as ex:
        logging.error("Column passed does not exist %s", str(ex))

    return new_df


def perform_feature_engineering(data_frame):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for
              naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']

    y_feat = data_frame['Churn']

    # X = pd.DataFrame()
    X[keep_cols] = data_frame[keep_cols]
    print(X.head())
    # This cell may take up to 15-20 minutes to run
    # train test split
    logging.info("Splitting data 70% train 30% test")
    return train_test_split(X, y_feat, test_size= 0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    try:
        assert y_train is not None
        assert y_test is not None
        assert y_train_preds_lr is not None
        assert y_train_preds_rf is not None
        assert y_test_preds_lr is not None
        assert y_test_preds_rf is not None
        plt.figure(figsize=(10, 10))
        #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
        plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties =
        'monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
        {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties =
        'monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
        {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig('C:/Work/Courses/Machine Learning DevOps Engineer/Predict customer churn/'+
        'images/results/rf_class_rpt.png')

        plt.figure(figsize=(10, 10))
        plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10},
        fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
        {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10},
        fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
        {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig('C:/Work/Courses/Machine Learning DevOps Engineer/Predict customer churn/'+
        'images/results/lr_class_rpt.png')

        logging.info("Reports save successfully as png at ./images/results")

    except AssertionError as ex_g:
        logging.error("Parameters cannot be null or none: %s ",ex_g)

def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    try:
        assert model is not None
        assert x_data is not None
        assert output_pth is not None
        # Calculate feature importances
        importances = model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        # Rearrange feature names so they match the sorted feature importances
        names = [x_data.columns[i] for i in indices]
        # Create plot
        plt.figure(figsize=(20,10))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(x_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(x_data.shape[1]), names, rotation=90)
        plt.savefig(output_pth+'feature_importance.png')
        print("RUnning save file at %s", output_pth+'feature_importance.png')
        logging.info("Saved importance feature at %s", output_pth+'feature_importance.png')
    except AssertionError:
        logging.error("Could not save feature importance plot. One or more values are None")

def tree_explainer_plot(model, x_test, output_pth):
    '''
    explain the model's predictions using SHAP values,
    creates and stores the tree explainer in pth
    input:
            model: model object containing tree explainer
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    try:
        assert model is not None
        assert x_test is not None
        assert output_pth is not None
        explainer = shap.TreeExplainer(model.best_estimator_)
        shap_values = explainer.shap_values(x_test)
        shap.summary_plot(shap_values, x_test, plot_type="bar")
        plt.savefig(output_pth+'tree_explainer.png')
        print("Saving file at %s", output_pth+'tree_explainer.png')
        logging.info("Saved importance feature at %s", output_pth+'tree_explainer.png')
    except AssertionError:
        logging.error("Could not save tree explainer plot, possibly one or more params is none")

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    try:
        ## TRAIN BOTH MODELS
        logging.info("Fitting Grid Search for a Random Forest model")
        cv_rfc.fit(X_train, y_train)
        logging.info("Fitting Logistic regression model")
        lrc.fit(X_train, y_train)

        ## STORE MODEL RESULTS : IMAGES
        lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    except Exception:
        logging.error("Models fitting failed")

    # plot both RF and LR roc
    plt.figure(figsize=(15, 8))
    ax_fig = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax_fig, alpha=0.8)
    lrc_plot.plot(ax=ax_fig, alpha=0.8)
    plt.show()
    logging.info("Saving models roc plot")
    plt.savefig('C:/Work/Courses/Machine Learning DevOps Engineer/Predict customer churn/'+
    'images/results/roc_plot.png')
    feature_importance_plot(cv_rfc, X,'C:/Work/Courses/Machine Learning DevOps Engineer/'+
    'Predict customer churn/images/results/')
    tree_explainer_plot(cv_rfc, X,'C:/Work/Courses/Machine Learning DevOps Engineer/'+
    'Predict customer churn/images/results/')
    # STORE MODEL SCORES

    logging.info("Saving models scores")
    savetxt('./logs/y_train_preds_lr.csv', lrc.predict(X_train), delimiter=',')
    savetxt('./logs/y_train_preds_rf.csv', cv_rfc.best_estimator_.predict(X_train), delimiter=',')
    savetxt('./logs/y_test_preds_lr.csv', lrc.predict(X_test), delimiter=',')
    savetxt('./logs/y_test_preds_rf.csv', cv_rfc.best_estimator_.predict(X_test), delimiter=',')
    ##
    # classification_report_image(y_train, y_test)
    # save best model
    logging.info("Saving best model")
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


def print_path():
    from pathlib import Path
    EDA_PATH = "images\eda"
    print(Path(EDA_PATH, "age_hist.png"))