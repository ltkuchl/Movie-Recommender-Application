
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# General: -------------------------------------------------------------------------------

def generate_rating_matrix(ratings_df):
    """
    Generate a user-item rating matrix from the ratings DataFrame.

    Parameters:
        ratings_df (pd.DataFrame): DataFrame containing 'userId', 'movieId', and 'rating' columns.

    Returns:
        pd.DataFrame: User-item rating matrix with users as rows, movies as columns, and ratings as values.
    """
    return ratings_df.pivot(index='userId', columns='movieId', values='rating')


# Pairwise similarity functions: ------------------------------------------------

def cosine_similarity_movies(rating_matrix):
    '''Compute cosine similarity between movies.'''
    review_matrix_t = rating_matrix.T
    movie_similarity = cosine_similarity(review_matrix_t.fillna(0)) # Note: unrated items contribute 
    return pd.DataFrame(movie_similarity, index=rating_matrix.columns, columns=rating_matrix.columns)

def cosine_similarity_users(rating_matrix):
    '''Compute cosine similarity between users.'''
    user_similarity = cosine_similarity(rating_matrix.fillna(0))  # Note: unrated items contribute
    return pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# Making predictions: ---------------------------------------------------------------

def predict_ratings_user_based(user_id, movie_id, ratings_matrix, user_similarity_df, k=10):
    """
    Predict a user's rating for a movie using collaborative filtering with k-nearest neighbors, i.e. by looking at how the k most similar who have watched the movie rated it. 
    Implements the mean centered version of user-based collaborative filtering.
    
    Parameters:
        user_id (int): ID of the target user
        movie_id (int): ID of the movie to predict rating for
        review_matrix (pd.DataFrame): Matrix of user-movie ratings where rows are users, columns are movies
        user_similarity_df (pd.DataFrame): Matrix of user-user similarities
        k (int, optional): Number of similar users to consider. Defaults to 10
    
    Returns:
        float: Predicted rating for the target movie by the target user
        
    Algorithm:
        1. Calculate target user's average rating
        2. Find k most similar users who rated the target movie
        3. Calculate rating offset for similar users (deviation from their mean rating)
        4. Predict rating as user's mean + weighted sum of neighbors' rating offsets
    """

    # calculate average rating of the target user
    user_reviews = ratings_matrix.loc[user_id]
    average_user_review = user_reviews[user_reviews != 0].mean() 

    # find k most similar users who have rated the target movie
    knn = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id) # most similar users excluding self
    knn = knn[ratings_matrix[movie_id] != 0]  # only users who have rated the target movie
    knn = knn.head(k)

    # get the rating offsets of these similar users for the target movie (their rating minus their average rating)
    all_neighbor_ratings = ratings_matrix.loc[knn.index] # all reviews made by neighbors
    movie_neighbor_ratings = all_neighbor_ratings[movie_id] # reviews for the target movie by neighbors
    neighbor_rating_offsets = movie_neighbor_ratings - all_neighbor_ratings[all_neighbor_ratings != 0].mean(axis=1)

    # calculate predicted rating
    nominator = np.sum(knn * neighbor_rating_offsets) # sum of similar user rating offsets weighted by similarity
    denominator = np.sum(knn) # sum of similar user similarities
    return average_user_review + (nominator / denominator)



def predict_rating_item_based(user_id, item_id, ratings_matrix, similarity_matrix, k=10):
    """
    Predicts the rating a user would give to an item using item-based collaborative filtering.

    Parameters
    ----------
    user_id : int or str
        The user for whom we want to predict the rating.
    item_id : int or str
        The target item for which we want a predicted rating.
    ratings_matrix : pd.DataFrame
        A user–item matrix where rows = users, columns = items, and entries = ratings.
        Unrated items should be NaN.
    similarity_matrix : pd.DataFrame
        An item–item similarity matrix with the same item labels as ratings_matrix columns.
    k : int
        The number of most similar items to consider (top-k neighbors).

    Returns
    -------
    float
        Predicted rating for (user_id, item_id). Returns np.nan if insufficient data.
    """


    # 0. Just in case set zeros to NaN for proper handling
    ratings_matrix = ratings_matrix.replace(0, np.nan)

    # 1. Get all items rated by the target user
    user_ratings = ratings_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings.notna()].index

    # 2. Get similarities between the target item and all items the user has rated
    sims = similarity_matrix.loc[item_id, rated_items]

    # 3. Select top-k most similar items
    top_k_items = sims.nlargest(k).index
    top_k_sims = sims.loc[top_k_items]
    top_k_ratings = user_ratings.loc[top_k_items]

    # 4. Compute weighted average (mean-centered version)
    # If no similar items, return NaN
    if top_k_sims.sum() == 0:
        return np.nan

    # Optional: mean-centering around user's mean
    user_mean = user_ratings.mean()

    # Weighted sum of deviations
    numerator = np.sum(top_k_sims * (top_k_ratings - user_mean))
    denominator = np.sum(np.abs(top_k_sims))

    prediction = user_mean + numerator / denominator

    return prediction
