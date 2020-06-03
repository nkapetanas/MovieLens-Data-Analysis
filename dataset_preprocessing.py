import pandas as pd

DATASET_PATH_MOVIES_CSV = "C:/Users/Delta/PycharmProjects/MovieLens-Data-Analysis/data//movies.csv"
DATASET_PATH_RATINGS_CSV = "C:/Users/Delta/PycharmProjects/MovieLens-Data-Analysis/data//ratings.csv"
DATASET_PATH_TAGS_CSV = "C:/Users/Delta/PycharmProjects/MovieLens-Data-Analysis/data//tags.csv"

def read_csv_file(file_path):
    return pd.read_csv(file_path)

def write_df_to_csv():
    with open('file.csv', 'a') as file:
        df.to_csv(file, header=True, index=False)

def movies_preprocessor(df):
    genres_df = df["genres"].str.split('|', expand=True)
    df["genres"] = genres_df.values.tolist()
    df["genres"] = df["genres"].apply(lambda el: [x for x in el if pd.notna(x)])
    return df


def ratings_preprocessor(df):
    pass

def tags_preprocessor(df):
    return df.groupby("movieId").agg({"tag": lambda x: list(x.str.lower())})


movies_df = read_csv_file(DATASET_PATH_MOVIES_CSV)
#ratings_df = read_csv_file(DATASET_PATH_RATINGS_CSV)
tags_df = read_csv_file(DATASET_PATH_TAGS_CSV)

movies_preprocessor(movies_df)
df = tags_preprocessor(tags_df)