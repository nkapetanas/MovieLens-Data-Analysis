import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(42)

DATASET_PATH_PREPROCESSED_CSV = "C:/Users/Delta/PycharmProjects/MovieLens-Data-Analysis/data/preprocessed_file.csv"
MAX_ITERATIONS = 1000


def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding="utf-8")


def create_bag_of_words(df, column):
    tfidf_ngram = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)
    tfidf_ngram.fit(df[column])
    trainq1_trans = tfidf_ngram.transform(df[column]).toarray()
    return np.concatenate((df["movieId"].values.reshape(-1,1), trainq1_trans), axis=1)


def jaccard_distance(list_a, list_b):
    try:
        return 1 - float(len(list_a.intersection(list_b))) / float(len(list_a.union(list_b)))
    except TypeError:
        print("Invalid type. Type set expected.")


def cosine_similarity(vec1, vec2):
    vec1 = [int(x) for x in vec1]
    vec2 = [int(x) for x in vec2]

    return (1 - spatial.distance.cosine(vec1, vec2))


class KMeans:

    def __init__(self, k, distance_function, plot_steps=False):
        self.K = k
        self.max_iterations = MAX_ITERATIONS
        self.plot_steps = plot_steps
        self.distance_function = distance_function

        # initialize clusters with empty list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, movies):
        self.number_of_records, self.features = movies.shape

        # initialize
        random_indexes = np.random.choice(self.number_of_records, self.K, replace=False)
        self.centroids = [movies[idx] for idx in random_indexes]

        # Optimize clusters
        for _ in range(self.max_iterations):

            # Assign samples to closest centroids (create clusters)
            self.clusters = self.create_clusters(self.centroids, movies)

            if self.plot_steps:
                self.plot(movies)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self.get_centroids(self.clusters, movies)

            # check if clusters have changed
            if self.converges(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot(movies)

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.number_of_records)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def create_clusters(self, centroids, movies):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, movie in enumerate(movies):
            centroid_index = self.get_closest_centroid(movie, centroids)
            clusters[centroid_index].append(idx)
        return clusters

    def get_closest_centroid(self, movies, centroids):
        # distance of the current sample to each centroid
        distances = [self.distance_function(movies, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def get_centroids(self, clusters, movies):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(movies[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def converges(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [self.distance_function(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self, movies):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = movies[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()


preprocessed_df = read_csv_file(DATASET_PATH_PREPROCESSED_CSV)


def get_movies_with_ratings(df):
    return df.iloc[:, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values


def get_movies_with_genres(df):
    return df.iloc[:, [0, 2]]


def get_movies_with_genres(df):
    return df.iloc[:, [0, 2]]


def get_movies_with_tags(df):
    return df.iloc[:, [0, 13]]


movies_with_ratings = get_movies_with_ratings(preprocessed_df)
# tags_transform = create_bag_of_words(get_movies_with_tags(preprocessed_df), "tag")
# genres_transform = create_bag_of_words(get_movies_with_genres(preprocessed_df), "genres")

k = KMeans(5, cosine_similarity)
y_pred = k.predict(movies_with_ratings)

k.plot(movies_with_ratings)