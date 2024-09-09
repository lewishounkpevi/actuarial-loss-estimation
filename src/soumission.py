import pickle
import pandas as pd
import time
import os
from data_preprocessing import preprocess_data, traiter_dataframe


def predict(df):

    # data prep

    ClaimNumber = df["ClaimNumber"]

    nlp_clusters = traiter_dataframe(df, "ClaimDescription", n_clusters=4)

    X_test, preprocessor = preprocess_data(df, nlp_clusters)

    # Prediction

    name = "/home/lewis/actuarial-loss-prediction/models/xgb_model_2024-09-09.pkl"

    with open(name, "rb") as f:
        loaded_model = pickle.load(f)

    # model = mlflow.pyfunc.load_model("models:/best_model/1")

    UltimateIncurredClaimCost = loaded_model.predict(X_test)

    submission = pd.DataFrame(
        {
            "ClaimNumber": ClaimNumber,  # Use ClaimNumber for submission
            "UltimateIncurredClaimCost": UltimateIncurredClaimCost,
        }
    )

    return submission


if __name__ == "__main__":

    test_df = pd.read_csv("data/input/test.csv")

    submission = predict(test_df)

    # Save the submission file

    today = time.strftime("%Y-%m-%d")
    name = os.getcwd() + "/" + "data/output" + "/" + "submission" + "_" + today + ".csv"

    submission.to_csv(name, index=False)

    print(f"Le fichier de soumission est stocké ici :{name}")
