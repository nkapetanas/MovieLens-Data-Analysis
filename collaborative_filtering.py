import numpy as np
from more_itertools import take
from scipy import spatial
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix
from collections import defaultdict

DATASET_PATH_MOVIES_CSV = "C:/Users/Delta/PycharmProjects/MovieLens-Data-Analysis/data/movies.csv"
DATASET_PATH_RATINGS_CSV = "C:/Users/Delta/PycharmProjects/MovieLens-Data-Analysis/data/ratings_temp.csv"


def cosine_similarity(vec1, vec2, indexes):
    similarity_indexes = dict()
    for userId in indexes:
        ratings_per_userid = vec2[userId, :].toarray()
        similarity_indexes[userId] = 1 - spatial.distance.cosine(vec1, ratings_per_userid)

    return similarity_indexes


def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding="utf-8")


def delete_df_column(df, column):
    del df[column]
    return df


def chunks(list_to_chunk, n):
    # looping till length l
    for i in range(0, len(list_to_chunk), n):
        yield list_to_chunk[i:i + n]


def sort_dict(dict_to_sort):
    sorted_dict = dict()
    for k in sorted(dict_to_sort, key=dict_to_sort.get, reverse=True):
        sorted_dict[k] = dict_to_sort[k]

    return sorted_dict


def get_n_elements_dict(n, reversed_list, dict_to_sort):
    dict_sort = dict()
    if reversed_list:
        for key in list(reversed(list(dict_to_sort)))[0:n]:
            dict_sort[key] = dict_to_sort[key]
        return dict_sort

    for key in list(dict_to_sort)[0:n]:
        dict_sort[key] = dict_to_sort[key]

    return dict_sort


def dictionary_to_sparse_matrix(input_dict):
    total_values = list()
    for _, value in input_dict.items():
        total_values.append(value[0])

    return csr_matrix(total_values)


def preprocess_dataset():
    movies_df = read_csv_file(DATASET_PATH_MOVIES_CSV)
    ratings_df = read_csv_file(DATASET_PATH_RATINGS_CSV)

    movies_df = delete_df_column(movies_df, "genres")
    movie_id_title_dict = movies_df.set_index("movieId").T.to_dict("list")

    ratings_df = delete_df_column(ratings_df, "timestamp")

    users = list(ratings_df["userId"].unique())
    movies = list(ratings_df["movieId"].unique())
    ratings = list(ratings_df["rating"])
    rows = ratings_df["userId"].astype(CategoricalDtype(categories=users)).cat.codes
    cols = ratings_df["movieId"].astype(CategoricalDtype(categories=movies)).cat.codes

    index_to_userId = {z: x for z, x in enumerate(users)}
    index_to_movieId = {z: x for z, x in enumerate(movies)}

    user_item = csr_matrix((ratings, (rows, cols)), shape=(len(users), len(movies)))

    cols = ratings_df["userId"].astype(CategoricalDtype(categories=users)).cat.codes
    rows = ratings_df["movieId"].astype(CategoricalDtype(categories=movies)).cat.codes

    item_user = csr_matrix((ratings, (rows, cols)), shape=(len(movies), len(users)))

    userid_movies_with_ratings = ratings_df.groupby("userId").agg(
        {"movieId": lambda x: list(x), "rating": lambda x: list(x)})

    userId_indexes = userid_movies_with_ratings.axes[0]

    userid_movies_with_ratings["userId"] = list(userid_movies_with_ratings.axes[0])

    ratings_df.groupby("userId").agg({"movieId": lambda x: list(x), "rating": lambda x: list(x)})

    return list(userId_indexes), users, movie_id_title_dict, user_item, item_user, movies, index_to_userId, index_to_movieId


def read_user_input(user_id_list):
    while True:
        try:
            collaborative_filtering_input = int(
                input("Please enter 0 for User-based, 1 for Item-based or 2 for Combination of the previous two: "))
            user_id_input = int(input("Give a User Id in order to present recommendations to that user: "))
        except ValueError:
            print('Please enter a valid number')

        if (collaborative_filtering_input in range(0, 3)) and (user_id_input in user_id_list):
            break
        else:
            print("Either the first choice is wrong or the user id does not exists, please try again")
    return collaborative_filtering_input, user_id_input


def user_based():

    user_indices = np.array(list(index_to_userId.keys()))
    list_of_chunks = list(chunks(user_indices, 100))

    rating_of_all_movies_active_user = user_item[user_id_input, :].toarray()
    all_movies_final_calculations = dict()

    for chunk in list(list_of_chunks):

        similar_users = cosine_similarity(rating_of_all_movies_active_user, user_item, chunk)

        twenty_closest_userIds = get_n_elements_dict(20, True, similar_users)
        top_twenty_user_ids = list(twenty_closest_userIds.keys())

        ratings_for_movies_for_top_users = dict()
        for userId in top_twenty_user_ids:
            ratings_for_movies_for_top_users[userId] = user_item[userId, :].toarray()

        indexes_per_user_rating_not_null = dict()
        for key, value in ratings_for_movies_for_top_users.items():
            indexes_per_user_rating_not_null[key] = ratings_for_movies_for_top_users[key][0].nonzero()

        sum_similarity_indexes = dict()
        for movie in movies:
            all_movies_final_calculations[movie] = 0
            sum_similarity_indexes[movie] = 0

        for key, value in indexes_per_user_rating_not_null.items():
            for movie in movies:
                if movie in value[0]:
                    sum_similarity_indexes[movie] = sum_similarity_indexes[movie] + similar_users[key]

        # multiply the similarity score with the ratings
        weighted_rating_matrix = dict()
        for key, value in similar_users.items():
            weighted_rating_matrix[key] = (user_item[key, :].toarray() * value)

        weighted_rating_sparse_matrix = dictionary_to_sparse_matrix(weighted_rating_matrix)
        sum_weighted_ratings_per_movie = weighted_rating_sparse_matrix.sum(axis=0)

        sum_weighted_ratings_per_movie = sum_weighted_ratings_per_movie[0].T
        for key, value in sum_similarity_indexes.items():
            if value != 0:
                sum_weighted_ratings_per_movie[key] = sum_weighted_ratings_per_movie[key] / value

        for i in range(0, len(sum_weighted_ratings_per_movie)):
            if sum_weighted_ratings_per_movie[i] != 0:
                all_movies_final_calculations[i] = sum_weighted_ratings_per_movie[i]

    top_twenty_movies_indexes_similarity = sort_dict(all_movies_final_calculations)
    n_movies = take(20, top_twenty_movies_indexes_similarity.items())

    return n_movies

def item_based():

    movie_indexes = np.array(list(index_to_movieId.keys()))
    all_movies_final_calculations = dict()

    list_of_chunks = list(chunks(movie_indexes, 100))
    movies_active_user = item_user[user_id_input, :].toarray()

    for chunk in list(list_of_chunks):

        similar_movies = cosine_similarity(movies_active_user, item_user, chunk)

        twenty_closest_movies = get_n_elements_dict(20, True, similar_movies)

        top_twenty_user_movies = list(twenty_closest_movies.keys())

        ratings_for_movies_for_top_users = dict()

        for movieId in top_twenty_user_movies:
            ratings_for_movies_for_top_users[movieId] = item_user[movieId, :].toarray()

        indexes_per_movie_rating_not_null = dict()
        for key, value in ratings_for_movies_for_top_users.items():
            indexes_per_movie_rating_not_null[key] = ratings_for_movies_for_top_users[key][0].nonzero()

        sum_similarity_indexes = dict()
        for movie in movies:
            all_movies_final_calculations[movie] = 0
            sum_similarity_indexes[movie] = 0

        for key, value in indexes_per_movie_rating_not_null.items():
            for movie in movies:
                if movie in value[0]:
                    sum_similarity_indexes[movie] = sum_similarity_indexes[movie] + similar_movies[key]

        # multiply the similarity score with the ratings
        weighted_rating_matrix = dict()
        for key, value in similar_movies.items():
            weighted_rating_matrix[key] = (item_user[key, :].toarray() * value)

        weighted_rating_sparse_matrix = dictionary_to_sparse_matrix(weighted_rating_matrix)
        sum_weighted_ratings_per_movie = weighted_rating_sparse_matrix.sum(axis=0)

        sum_weighted_ratings_per_movie = sum_weighted_ratings_per_movie[0].T
        for key, value in sum_similarity_indexes.items():
            if value != 0:
                sum_weighted_ratings_per_movie[key] = sum_weighted_ratings_per_movie[key] / value

        for i in range(0, len(sum_weighted_ratings_per_movie)):
            if sum_weighted_ratings_per_movie[i] != 0:
                all_movies_final_calculations[i] = sum_weighted_ratings_per_movie[i]

    top_twenty_movies_indexes_similarity = sort_dict(all_movies_final_calculations)
    n_movies = take(20, top_twenty_movies_indexes_similarity.items())

    return n_movies

userId_indexes, users, movie_id_title_dict, user_item, item_user, movies, index_to_userId, index_to_movieId = preprocess_dataset()
collaborative_filtering_input, user_id_input = read_user_input(users)

if collaborative_filtering_input == 0:
    n_movies = user_based()

    print("Top-20 movie recommendations for the proposed user are: \n")
    for movie in n_movies:
        print(str(movie_id_title_dict[movie[0]][0]) + " with similarity: " + str(movie[1]))

elif collaborative_filtering_input == 1:

    n_movies = item_based()
    print("Top-20 movie recommendations for the proposed user are: \n")
    for movie in n_movies:
        print(str(movie_id_title_dict[index_to_movieId[movie[0]]][0]) + " with similarity: " + str(movie[1]))

elif collaborative_filtering_input == 2:
    n_user_based_movies = user_based()
    n_item_based_movies = item_based()
    concatenated_results = n_user_based_movies + n_item_based_movies
    final_movie_results = defaultdict()
    for result in concatenated_results:
        final_movie_results[result[0]] = result[1]

    n_movies = sort_dict(final_movie_results)
    n_movies = take(20, n_movies.items())

    print("Top-20 movie recommendations for the proposed user are: \n")
    for movie in n_movies:
        print(str(movie_id_title_dict[index_to_movieId[movie[0]]][0]) + " with similarity: " + str(movie[1]))

else:
    print("Try run the program again with correct values")


