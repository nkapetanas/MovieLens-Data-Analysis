import copy
import math
import pandas as pd
import numpy as np
from random import random
from sklearn.feature_extraction.text import TfidfVectorizer
from random import seed
from random import randrange

DATASET_PATH_PREPROCESSED_CSV = "C:/Users/Delta/PycharmProjects/MovieLens-Data-Analysis/data/preprocessed_file.csv"
MAX_ITERATIONS = 1000


# Split a dataset into k folds
def cross_validation_split(dataset, folds=5):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def jaccard_distance(list_a, list_b):
    try:
        return 1 - float(len(list_a.intersection(list_b))) / float(len(list_a.union(list_b)))
    except TypeError:
        print("Invalid type. Type set expected.")


def get_cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0

    return float(numerator) / denominator


def create_bag_of_words(df, column):
    tfidf_ngram = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)
    tfidf_ngram.fit(df[column])
    trainq1_trans = tfidf_ngram.transform(df[column].values)
    return trainq1_trans


def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding="utf-8")


class K_Means():
    """
    k: int , number of clusters
    seed: int, will be randomly set if None
    max_iter: int, number of iterations to run algorithm, default: 200
    centroids: array, k, number_features
    cluster_labels: label for each data point
    """

    def __init__(self, k, distance_function, movies):
        self.k = k
        self.distance_function = distance_function

        self.movies = movies
        self.cluster_to_movie_ids = dict()  # cluster to movieID
        self.movie_ids_to_cluster = dict()  # reverse index, movieID to cluster
        self.jaccard_matrix = dict()  # stores pairwise jaccard distance in a matrix
        self.cosine_similarity_matrix = dict()  # stores pairwise cosine distance in a matrix

        self.seeds = self.seed_initialization()
        self.centroid_initialization()
        self.matrix_initialization()

    def split_text(self, input):
        pass

    # 2. Find the Jaccard/Cosine distance of each point in the data set
    # with the identified K points — cluster centroids.
    def matrix_initialization(self):
        # creates matrix storing pairwise jaccard distances
        for movie_id in self.movies:
            self.jaccard_matrix[movie_id] = dict()
            bag1 = set(self.split_text(self.movies[movie_id]['text']))

            for movie_id2 in self.movies:
                if movie_id2 not in self.jaccard_matrix:
                    self.jaccard_matrix[movie_id2] = dict()
                bag2 = set(self.split_text(self.movies[movie_id2]['text']))

                distance = self.distance_function(bag1, bag2)
                self.jaccard_matrix[movie_id][movie_id2] = distance
                self.jaccard_matrix[movie_id2][movie_id] = distance

    def centroid_initialization(self):
        # Initialize movies to no cluster
        for movie_id in self.movies:
            self.movie_ids_to_cluster[movie_id] = -1

        # Initialize clusters with seeds
        for k in range(self.k):
            self.cluster_to_movie_ids[k] = {self.seeds[k]}
            self.movie_ids_to_cluster[self.seeds[k]] = k

    # 1. Pick K points as the initial centroids from the data set, either randomly or the first K.
    def seed_initialization(self):
        # Computes initial seeds for k-means using k-means++ algorithm

        # 1. Choose one center uniformly at random from among the data points
        seed = random.choice(self.movies.keys())

        # 2. For each data point x, compute distance(x),
        # the distance between x and the nearest center that has already been chosen
        seeds = {seed}

        while len(seeds) < self.k:
            distance_matrix = {}
            sum_of_distance = 0

            for seed in seeds:
                bag1 = set(self.split_text(self.movies[seed]['text']))

                for movie_id in self.movies:
                    if movie_id == seed:
                        continue
                    bag2 = set(self.split_text(self.movies[movie_id]['text']))
                    calculated_distance = self.distance_function(bag1, bag2)

                    if movie_id not in distance_matrix or calculated_distance < distance_matrix[movie_id]:
                        distance_matrix[movie_id] = calculated_distance

            prob_dict = dict()
            for movie_id in distance_matrix:
                sum_of_distance += np.power(distance_matrix[movie_id], 2)

            for movie_id in distance_matrix:
                prob_dict[movie_id] = np.power(distance_matrix[movie_id], 2) / sum_of_distance

            # 3. Choose one new data point at random as a new center, using a weighted probability distribution
            # where a point x is chosen with probability proportional to D(x)^2.
            movie_ids, weights = prob_dict.keys(), prob_dict.values()
            seed = random.choice(movie_ids, p=weights)
            seeds.add(seed)

        # 4. Repeat Steps 2 and 3 until k centers have been chosen.
        return list(seeds)

    # 3. Assign each data point to the closest centroid using the distance found in the previous step.
    def calculate_new_cluster(self):
        new_cluster_to_movieIds = dict()
        new_movieIds_to_cluster = dict()

        for k in range(self.k):
            new_cluster_to_movieIds[k] = set()

        for movie_id in self.movies:
            minimum_distance = float("inf")
            min_cluster = self.movie_ids_to_cluster[movie_id]

            # Calculate min average distance to each cluster
            for k in self.cluster_to_movie_ids:
                dist = 0
                count = 0
                for movie_id_cluster in self.cluster_to_movie_ids[k]:
                    dist += self.jaccard_matrix[movie_id][movie_id_cluster]
                    count += 1
                if count > 0:
                    avg_dist = dist / float(count)
                    if minimum_distance > avg_dist:
                        minimum_distance = avg_dist
                        min_cluster = k

            new_cluster_to_movieIds[min_cluster].add(movie_id)
            new_movieIds_to_cluster[movie_id] = min_cluster

        return new_cluster_to_movieIds, new_movieIds_to_cluster

    # 4. Find the new centroid by taking the average of the points in each cluster group.
    def update_centroids(self):

        # Initialize previous cluster to compare changes with new clustering
        new_cluster_to_movieIds, new_movieIds_to_clusters = self.calculate_new_cluster()

        self.cluster_to_movie_ids = copy.deepcopy(new_cluster_to_movieIds)
        self.movie_ids_to_cluster = copy.deepcopy(new_movieIds_to_clusters)

        # Converges until old and new iterations are the same
        iterations = 1
        while iterations < MAX_ITERATIONS:
            new_cluster_to_movieIds, new_movieIds_to_clusters = self.calculate_new_cluster()
            iterations += 1

            if self.movie_ids_to_cluster != new_movieIds_to_clusters:
                self.cluster_to_movie_ids = copy.deepcopy(new_cluster_to_movieIds)
                self.movie_ids_to_cluster = copy.deepcopy(new_movieIds_to_clusters)
            return

    def print_clusters(self):
        # Prints cluster ID and movie IDs for that cluster
        for k in self.movie_ids_to_cluster:
            print(str(k) + ':' + ','.join(map(str, self.movie_ids_to_cluster[k])))


preprocessed_df = read_csv_file(DATASET_PATH_PREPROCESSED_CSV)
# tags_transform = create_bag_of_words(preprocessed_df, "tag")
# genres_transform = create_bag_of_words(preprocessed_df, "genres")

k_means = K_Means(5, jaccard_distance, preprocessed_df)
k_means.update_centroids()
k_means.print_clusters()
