"""Recommender system module."""

import numpy as np
import pandas as pd

class CollaborativeFilter:
    """Item-based collaborative filter class."""

    def __init__(self, ratings: pd.DataFrame, min_ratings: int = 10):
        """Initialize the CollaborativeFilter class.

        Args:
            ratings (pd.DataFrame): User-movie ratings DataFrame.
            min_ratings (int): Minimum number of ratings for a movie to be considered.
        """
        self._min_ratings = min_ratings

        # Filter out movies with fewer than min_ratings
        rating_count = ratings.groupby('movieId').size() # Count the number of ratings for each movie
        valid_movies = rating_count[rating_count >= self._min_ratings].index # Get movie IDs with enough ratings
        new_ratings = ratings[ratings['movieId'].isin(valid_movies)] 

        # Create a user-movie matrix: rows are movies, columns are users, and values are ratings
        matrix = new_ratings.pivot_table(index='movieId', columns='userId', values='rating', fill_value=0)

        # Divide each row by its geometric length to normalize the movie vectors
        row_lengths = np.linalg.norm(matrix.to_numpy(), axis=1, keepdims=True) 
        self._movie_vectors = matrix / row_lengths

    def recommend(self, user_ratings: dict[int, float], n: int = 5) -> pd.Series:
        """Recommend movies based on user ratings.

        Args:
            user_ratings (dict[int, float]): User ratings for movies.
            n (int): Number of recommendations to return.

        Returns:
            pd.Series: Recommended movies with their predicted ratings.
        """
        raise NotImplementedError("The recommend method is not implemented yet.")

    def _similarities(self, movie_ids: list[int]) -> pd.DataFrame:
        """Compute the similarity scores between the given movies and all other movies.

        Args:
            movie_ids (list[int]): List of movie IDs to compute similarities for.

        Returns:
            pd.DataFrame: DataFrame containing similarity scores between the given movies and all other movies.
        """
        # Filter out movie IDs that are not in the movie vectors
        valid_movie_ids = [movie_id for movie_id in movie_ids if movie_id in self._movie_vectors.index]

        # Compute the dot product of the movie vectors for the valid movie IDs
        return self._movie_vectors.loc[valid_movie_ids].dot(self._movie_vectors.T)
    
    @staticmethod
    def _weighted_sum(similarities: pd.DataFrame, user_ratings: dict[int, float]) -> pd.Series:
        """Compute weighted sum of ratings based on similarities.

        Args:
            similarities (pd.DataFrame): DataFrame containing similarity scores.
            user_ratings (dict[int, float]): User ratings for movies.

        Returns:
            pd.Series: Weighted sum of ratings for each movie.
        """
        raise NotImplementedError("The _weighted_sum method is not implemented yet.")
    
    def breakdown(self, user_ratings: dict[int, float], movie_id: int) -> pd.Series:
        """Compute how much each of the user's ratings contributed to one movie's recommendation.

        Args: 
            user_ratings (dict[int, float]): User ratings for movies.
            movie_id (int): Movie ID for which to compute the breakdown.
        
        Returns:
            pd.Series: Contribution of each rated movie to the recommendation of the specified movie.
        """
        raise NotImplementedError("The breakdown method is not implemented yet.")
