import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from scipy.stats import sem, t
import joblib

RANDOM_STATE = 42

def build_pipeline(regressor, feature_selection=None, k=10):
    """
    Builds a machine learning pipeline with optional feature selection and scaling.

    Parameters:
        regressor: The regression model to be used in the pipeline.
        feature_selection (str, optional): The feature selection method. 'selectkbest' or 'rfe'. Defaults to None.
        k (int, optional): The number of features to select. Defaults to 10.

    Returns:
        Pipeline: The constructed machine learning pipeline.
    """
    steps = [('scaler', RobustScaler())]
    if feature_selection == 'selectkbest':
        steps.append(('feature_selection', SelectKBest(score_func=f_regression, k=k)))
    elif feature_selection == 'rfe':
        steps.append(('feature_selection', RFE(estimator=ElasticNet(random_state=RANDOM_STATE), n_features_to_select=k)))
    steps.append(('regressor', regressor))
    return Pipeline(steps)

def evaluate_model(model, X, y, n_splits=5, n_repeats=1, compute_statistics=True):
    """
    Evaluates a regression model using repeated k-fold cross-validation.

    Parameters:
        model: The trained regression model.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        n_splits (int, optional): The number of folds. Defaults to 5.
        n_repeats (int, optional): The number of repetitions. Defaults to 1.
        compute_statistics (bool, optional): Whether to compute and return statistics. Defaults to True.

    Returns:
        tuple: A tuple containing a DataFrame of results and a dictionary of statistics (if compute_statistics is True).
               If compute_statistics is False, only the DataFrame of results is returned.
    """
    results = []
    kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    for train_index, test_index in kf.split(X):
        model.fit(X.iloc[train_index], y.iloc[train_index])
        y_pred = model.predict(X.iloc[test_index])
        results.append({
            "RMSE": np.sqrt(mean_squared_error(y.iloc[test_index], y_pred)),
            "MAE": mean_absolute_error(y.iloc[test_index], y_pred),
            "R2": r2_score(y.iloc[test_index], y_pred)
        })
    results_df = pd.DataFrame(results)
    if compute_statistics:
        statistics = {}
        for metric in ["RMSE", "MAE", "R2"]:
            metric_values = results_df[metric]
            mean_val = metric_values.mean()
            median_val = metric_values.median()
            ci_low, ci_high = compute_confidence_interval(metric_values)
            statistics[metric] = {
                "mean": mean_val,
                "median": median_val,
                "95% CI": (ci_low, ci_high)
            }
        return results_df, statistics
    return results_df

def compute_confidence_interval(data, confidence=0.95):
    """
    Computes the confidence interval for a given dataset.

    Parameters:
        data (pd.Series): The dataset.
        confidence (float, optional): The confidence level. Defaults to 0.95.

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    n = len(data)
    mean_val = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean_val - h, mean_val + h

def select_features(X, y, method='selectkbest', k=10):
    """
    Selects features from the feature matrix using either SelectKBest or RFE.

    Parameters:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        method (str, optional): The feature selection method. 'selectkbest' or 'rfe'. Defaults to 'selectkbest'.
        k (int, optional): The number of features to select. Defaults to 10.

    Returns:
        tuple: A tuple containing the transformed feature matrix and the list of selected features.
    """
    if method == 'selectkbest':
        selector = SelectKBest(score_func=f_regression, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
    elif method == 'rfe':
        selector = RFE(estimator=ElasticNet(random_state=RANDOM_STATE), n_features_to_select=k)
        selector.fit(X, y)
        selected_features = X.columns[selector.support_]
        X_new = X[selected_features]
    else:
        raise ValueError("Invalid feature selection method. Choose 'selectkbest' or 'rfe'.")
    print(f"✅ Selected features: {list(selected_features)}")
    return X_new, selected_features

def tune_hyperparameters(X, y, regressor, param_grid, cv_folds=5):
    """
    Tunes the hyperparameters of a regression model using GridSearchCV.

    Parameters:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        regressor: The regression model pipeline.
        param_grid (dict): The parameter grid for hyperparameter tuning.
        cv_folds (int, optional): The number of cross-validation folds. Defaults to 5.

    Returns:
        Pipeline: The best estimator found by GridSearchCV.
    """
    pipeline = regressor # The regressor here is already a pipeline
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=cv_folds,
        n_jobs=-1,
        verbose=1
    )
    print("🔍 Starting hyperparameter tuning...")
    grid_search.fit(X, y)
    print(f"✅ Best hyperparameters: {grid_search.best_params_}")
    print(f"✅ Best RMSE: {-grid_search.best_score_}")
    return grid_search.best_estimator_

def save_model(model, model_name, directory="./models"):
    """
    Saves a trained model to a joblib file.

    Parameters:
        model: The trained model.
        model_name (str): The name of the model.
        directory (str, optional): The directory to save the model. Defaults to "./models".
    """
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")

def load_model(model_path):
    """
    Loads a trained model from a joblib file.

    Parameters:
        model_path (str): The path to the joblib file.

    Returns:
        object: The loaded model.
    """
    return joblib.load(model_path)

def visualize_results(evaluation_results):
    """
    Visualizes the evaluation results using boxplots.

    Parameters:
        evaluation_results (dict): A dictionary containing the evaluation results for each model stage and regressor.
    """
    for metric in ["RMSE", "MAE", "R2"]:
        plt.figure(figsize=(12, 6))  # Increased figure size for better readability
        data_frames = []
        for stage, regressor_results in evaluation_results.items():
            for regressor, df in regressor_results.items():
                df['Regressor'] = regressor
                df['Stage'] = stage
                data_frames.append(df)
        
        combined_df = pd.concat(data_frames, ignore_index=True)
        sns.boxplot(x="Stage", y=metric, hue="Regressor", data=combined_df)
        plt.title(f"{metric} Comparison Across Models and Regressors")
        plt.ylabel(metric)
        plt.xlabel("Model Stage")
        plt.tight_layout()
        plt.show()