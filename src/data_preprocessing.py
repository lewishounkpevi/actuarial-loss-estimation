import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import string
import numpy as np


# Téléchargement des données nécessaires pour nltk
nltk.download("punkt_tab")
nltk.download("stopwords")


# Fonction de nettoyage du texte
def nettoyer_texte(text):
    # Convertir le texte en minuscules
    text = text.lower()

    # Supprimer la ponctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Supprimer les stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Retourner les tokens nettoyés
    return tokens


# Fonction pour convertir une colonne de texte en vecteurs Word2Vec
def text_to_vectors(text_data):
    # Nettoyage des textes
    text_data = text_data.apply(nettoyer_texte)

    # Entraînement du modèle Word2Vec
    model = Word2Vec(
        sentences=text_data, vector_size=100, window=5, min_count=1, workers=4
    )

    # Moyenne des vecteurs pour chaque texte
    def vectorize_text(tokens):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(100)

    # Application de la vectorisation sur chaque ligne de texte
    text_vectors = text_data.apply(vectorize_text)
    return text_vectors


# Fonction de clustering et étiquetage
def clustering_texts(vectors, n_clusters=5):
    # Convertir la liste de vecteurs en une matrice numpy
    X = np.vstack(vectors)

    # Appliquer KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=5)
    labels = kmeans.fit_predict(X)

    # Calculer le score de silhouette
    silhouette_avg = silhouette_score(X, labels)
    print(
        f"Le score moyen de silhouette pour {n_clusters} clusters est : {silhouette_avg:.3f}"
    )

    return labels


# Exemple d'utilisation avec une dataframe Pandas
def traiter_dataframe(df, text_column, n_clusters=5):
    # Convertir la colonne de texte en vecteurs
    text_vectors = text_to_vectors(df[text_column])

    # Appliquer le clustering
    labels = clustering_texts(text_vectors, n_clusters)

    # Ajouter les étiquettes de clusters à la dataframe
    df["Cluster_Label"] = labels

    return df


def preprocess_data(train_df, nlp_clusters):
    """
    Preprocess the input dataframe with feature engineering and transformations.

    Parameters:
    - train_df: pd.DataFrame, the training data
    - nlp_clusters: pd.DataFrame, containing 'Cluster_Label' to add to the train_df

    Returns:
    - X_train, X_val, y_train, y_val: Preprocessed training and validation sets
    - preprocessor: The fitted preprocessor for future transformations
    """
    # Drop unnecessary columns
    train_df.drop(["ClaimNumber", "ClaimDescription"], axis=1, inplace=True)

    # Convert Date columns to datetime
    train_df["DateTimeOfAccident"] = pd.to_datetime(
        train_df["DateTimeOfAccident"], errors="coerce"
    )
    train_df["DateReported"] = pd.to_datetime(train_df["DateReported"], errors="coerce")

    # Feature engineering: calculate the delay between accident and report
    train_df["AccidentReportDelay"] = (
        train_df["DateReported"] - train_df["DateTimeOfAccident"]
    ).dt.days
    train_df.drop(["DateTimeOfAccident", "DateReported"], axis=1, inplace=True)

    # Handle missing values for 'MaritalStatus'

    train_df["MaritalStatus"].replace("U", "S", inplace=True)

    train_df["Gender"].replace("U", "M", inplace=True)

    # Add Cluster Label from NLP processing
    train_df["Cluster_Label"] = nlp_clusters["Cluster_Label"]

    # Define categorical and numerical columns
    categorical_columns = [
        "Gender",
        "MaritalStatus",
        "PartTimeFullTime",
        "Cluster_Label",
    ]
    numerical_columns = [
        "Age",
        "DependentChildren",
        "DependentsOther",
        "WeeklyWages",
        "HoursWorkedPerWeek",
        "DaysWorkedPerWeek",
        "AccidentReportDelay",
        "InitialIncurredCalimsCost",
    ]

    X = train_df[categorical_columns + numerical_columns]

    # Preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    # Fit and transform the training data, and transform validation data
    preprocessed_X = preprocessor.fit_transform(X)

    return preprocessed_X, preprocessor


if __name__ == "__main__":
    # Example usage with sample train_df and nlp_clusters
    train_df1 = pd.read_csv("data/input/train.csv")

    train_df = train_df1.copy()
    # application nlp + clustering pour traiter la description des sinistres

    nlp_clusters = traiter_dataframe(train_df, "ClaimDescription", n_clusters=4)

    X, preprocessor = preprocess_data(train_df, nlp_clusters)

    print(f"Training data shape: {X.shape}")

    print("***** All Good ! *****")
