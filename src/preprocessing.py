from typing import Tuple
import re 
from pathlib import Path 
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

DATA_FOLDER = models_dir = Path(__file__).parent / ".." / "data"


def normalize_text(lst):
    return [re.sub(r"[^a-z\s]", "", re.sub(r"[\/,&\-]", " ", item.strip().lower())) for item in lst]

def get_data() -> list:
    print("fetching available data files...")
    all_files = os.listdir(DATA_FOLDER)
    movie_files = [(str(DATA_FOLDER) + '/') + file for file in all_files if file.startswith("movies_")]
    print("Files found: ")
    for file in movie_files:
        print(file + "\n")
    return movie_files

def load_all_data(movie_files) -> pd.DataFrame:
    print("Loading data...")
    dfs = []
    for file in movie_files:
        print(f"Loading {file}...")
        temp_df = pd.read_csv(file)
        dfs.append(temp_df)

    print("All files loaded!")
    print("Creating dataset...")

    movies = pd.concat(dfs, ignore_index=True)
    movies = movies.reset_index()
    print("Dataset created sucessfully!")
    return movies

def dedup_data(movies: pd.DataFrame) -> pd.DataFrame:
    print("Deduping data...")
    print(f"Original length: {len(movies)}")
    movies_deduped = movies.drop_duplicates()
    movies_filtered = movies_deduped.dropna(subset = ["genres", "overview"])
    print(f"Filtered length: {len(movies_filtered)}")
    return movies_filtered

def normalize_genres(movies: pd.DataFrame) -> pd.DataFrame:
    print("Normalizing genre data...")
    movies["genres"] = movies.genres.str.split("|")
    movies["genre_list"] = movies["genres"].apply(normalize_text)
    return movies

def create_labels(movies: pd.DataFrame) -> Tuple[np.ndarray,list]:
    print("Creating genre matrix...")
    genres = movies["genre_list"]
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies["genre_list"])

    return genre_matrix, mlb.classes_

def prepare_features(movies: pd.DataFrame) -> np.ndarray:
    print("Preparing summaries...")
    movies["overview"] = movies["overview"].str.lower().str.replace(r'[^a-z\s]','',regex=True)
    features = movies["overview"].to_numpy()
    return features

def split_data(X: np.ndarray,
            y: np.ndarray, 
            random_state: int = 42,
            test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                        random_state = random_state,
                                        test_size = test_size)
    
    return X_train, X_test, y_train, y_test