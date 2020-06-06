import pandas as pd
from nltk.corpus import stopwords

DATASET_PATH_MOVIES_CSV = "C:/Users/Delta/PycharmProjects/MovieLens-Data-Analysis/data/movies.csv"
DATASET_PATH_RATINGS_CSV = "C:/Users/Delta/PycharmProjects/MovieLens-Data-Analysis/data/ratings.csv"
DATASET_PATH_TAGS_CSV = "C:/Users/Delta/PycharmProjects/MovieLens-Data-Analysis/data/tags.csv"

TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
stop = set(stopwords.words('english'))

def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding="utf-8")

def write_df_to_csv(df):
    with open('data/preprocessed_file.csv', "w", encoding="utf-8") as file:
        df.to_csv(file, header=True, index=False)

def clean_data(df, column):
    df[column] = df[column].str.lower()
    df[column] = df[column].str.replace('[^\w\s]', '')
    df[column] = df[column].str.replace('<[^<]+?>', '')
    df[column] = df[column].apply(
        lambda x: ' '.join([item for item in x.split() if item not in stop]))
    return df

def movies_preprocessor(df):
    genres_df = df["genres"].str.split('|', expand=True)
    df["genres"] = genres_df.values.tolist()
    df["genres"] = df["genres"].apply(lambda el: [x for x in el if pd.notna(x)]).apply(' '.join)
    clean_data(df, "genres")
    df["genres"] = df["genres"].replace({"(no genres listed)": "Not-Applicable"})
    return df

def ratings_preprocessor(df):
    return df.groupby("movieId")["rating"].value_counts().to_frame().unstack("rating").fillna(float(0))


def tags_preprocessor(df):
    temp_df = df.groupby("movieId").agg({"tag": lambda x: list(x)})

    temp_df["tag"] = temp_df["tag"].apply(lambda x: ' '.join(map(str, x)))
    temp_df = clean_data(temp_df, "tag")
    return temp_df


movies_df = read_csv_file(DATASET_PATH_MOVIES_CSV)
ratings_df = read_csv_file(DATASET_PATH_RATINGS_CSV)
tags_df = read_csv_file(DATASET_PATH_TAGS_CSV)

ratings_per_movie = ratings_preprocessor(ratings_df)
genres_per_movie = movies_preprocessor(movies_df)
tags_per_movie = tags_preprocessor(tags_df)


concatenated_df = pd.concat([genres_per_movie, ratings_per_movie, tags_per_movie], axis=1)
concatenated_df["tag"] = concatenated_df["tag"].fillna("Not-applicable")
concatenated_df = concatenated_df.fillna(0)

# corrective actions
concatenated_df.movieId = concatenated_df.movieId.astype(int)
concatenated_df = concatenated_df[concatenated_df.movieId != 0]
concatenated_df = concatenated_df.dropna(subset=['movieId'])

write_df_to_csv(concatenated_df)
