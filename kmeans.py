from random import sample

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import spatial
from scipy.spatial.distance import cosine, jaccard, cdist
import seaborn as sns
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(42)

DATASET_PATH_PREPROCESSED_CSV = "C:/Users/Delta/PycharmProjects/MovieLens-Data-Analysis/data/preprocessed_file.csv"
MAX_ITERATIONS = 1000


def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding="utf-8")


def create_bag_of_words(df, column):
    tfidf_ngram = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=2000)
    tfidf_ngram.fit(df[column])
    trainq1_trans = tfidf_ngram.transform(df[column]).toarray()
    return np.concatenate((df["movieId"].values.reshape(-1, 1), trainq1_trans), axis=1)


def jaccard_distance_genres(list_a, list_b):
    return cdist(np.vstack(list_a), np.vstack(list_b), 'jaccard')

def jaccard_distance_tags(list_a, list_b):
    return cdist(np.vstack(list_a), np.vstack(list_b), 'jaccard')

def cosine_similarity(vec1, vec2):
    vec1 = [int(x) for x in vec1]
    vec2 = [int(x) for x in vec2]

    return (1 - spatial.distance.cosine(vec1, vec2))

def custom_distance_function(vec1, vec2):
    return 0.3*jaccard_distance_genres(vec1, vec2) + 0.25*jaccard_distance_tags(vec1, vec2) + 0.45*cosine_similarity(vec1, vec2)

class KMeans:

    def __init__(self, k, distance_function, plot_steps=False):
        self.K = k
        self.max_iterations = MAX_ITERATIONS
        self.plot_steps = plot_steps
        self.distance_function = distance_function

        self.clusters = [[] for _ in range(self.K)]

        self.centroids = []

    def predict(self, movies):
        self.number_of_records, self.features = movies.shape

        random_indexes = np.random.choice(self.number_of_records, self.K, replace=False)
        self.centroids = [movies[idx] for idx in random_indexes]

        for _ in range(self.max_iterations):

            self.clusters = self.create_clusters(self.centroids, movies)

            if self.plot_steps:
                self.plot(movies)

            centroids_old = self.centroids
            self.centroids = self.get_centroids(self.clusters, movies)

            if self.converges(centroids_old, self.centroids).all():
                break

            if self.plot_steps:
                self.plot(movies)


        return self.get_labels_of_clusters(self.clusters)

    def get_labels_of_clusters(self, clusters):

        labels = np.empty(self.number_of_records)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def create_clusters(self, centroids, movies):

        clusters = list()
        for _ in range(self.K):
            clusters.append(list())

        for index, movie in enumerate(movies):
            centroid_index = self.get_closest_centroid(movie, centroids)
            clusters[centroid_index].append(index)
        return clusters

    def get_closest_centroid(self, movies, centroids):
        # distance of the current sample to each centroid
        distances = list()
        for centroid in centroids:
            distances.append(self.distance_function(movies, centroid))

        closest_index = np.argmin(distances)
        return closest_index

    def get_centroids(self, clusters, movies):

        centroids = np.zeros((self.K, self.features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(movies[cluster], axis=0)
            centroids[cluster_index] = cluster_mean
        return centroids

    def converges(self, centroids_old, centroids):

        distances = list()
        for i in range(self.K):
            distances.append(self.distance_function(centroids_old[i], centroids[i]))

        return sum(distances) == 0

    def plot(self, movies):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = movies[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()


def get_movies_with_ratings(df):
    return df.iloc[:, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values


def get_movies_with_genres(df):
    return df.iloc[:, [0, 2]]


def get_movies_with_genres(df):
    return df.iloc[:, [0, 2]]


def get_movies_with_tags(df):
    return df.iloc[:, [0, 13]]


predictions_list = list()
for chunk in pd.read_csv(DATASET_PATH_PREPROCESSED_CSV, chunksize=1000):


    movies_with_ratings = get_movies_with_ratings(chunk)
    # tags_transform = create_bag_of_words(get_movies_with_tags(chunk), "tag")
    # genres_transform = create_bag_of_words(get_movies_with_genres(chunk), "genres")

    k = KMeans(5, cosine_similarity)
    y_pred = k.predict(movies_with_ratings)
    predictions_list.append(y_pred)

flat_list = [item for sublist in predictions_list for item in sublist]

