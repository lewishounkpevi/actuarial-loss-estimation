from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import time
import os
import pickle


def train_xgboost_with_gridsearch(train_df, X):
    """
    Train an XGBRegressor using GridSearchCV to find the optimal hyperparameters.
    RMSE is used as the performance indicator.

    Parameters:

    - train_df: Training Dataset
    - X: Preprocessed training features


    Returns:
    - best_model: The best model found by GridSearchCV
    - best_params: The best parameters found by GridSearchCV
    - rmse: Root Mean Squared Error on the validation set
    """

    y = train_df["UltimateIncurredClaimCost"]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model
    model = XGBRegressor(objective="reg:squarederror", random_state=42)

    # Define the parameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5],
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    # Fit the model using GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # # # Predict on validation set
    # y_train_pred = best_model.predict(X_train)

    # # Calculate Root Mean Squared Error (RMSE)
    # rmse_train = np.sqrt(mean_squared_error(X_train, y_train_pred))

    # Predict on validation set
    y_val_pred = best_model.predict(X_val)

    # Calculate Root Mean Squared Error (RMSE)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

    return best_model, grid_search.best_params_, rmse_val


def log_model_local(model, model_name):
    today = time.strftime("%Y-%m-%d")
    name = os.getcwd() + "/" + "models" + "/" + model_name + "_" + today + ".pkl"
    with open(name, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    
    from data_preprocessing import preprocess_data, traiter_dataframe

    train_df = pd.read_csv("data/input/train.csv")
    # application nlp + clustering pour traiter la description des sinistres

    nlp_clusters = traiter_dataframe(train_df, "ClaimDescription", n_clusters=4)

    X_train, preprocessor = preprocess_data(train_df, nlp_clusters)

    best_model, best_params, rmse_val = train_xgboost_with_gridsearch(train_df, X_train)

    log_model_local(best_model, "xgb_model")

    print(f"Best Parameters: {best_params}")
    # print(f"train RMSE: {rmse_train}")
    print(f"Validation RMSE: {rmse_val}")
