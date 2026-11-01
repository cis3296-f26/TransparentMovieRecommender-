import numpy as np
import pandas as pd
import pytest
from recommender import CollaborativeFilter

TEST_RATINGS = pd.DataFrame([(1, 10, 5.0), (2, 10, 3.0), (1, 20, 3.0), (2, 20, 5.0), (1, 30, 4.0)],
                               columns=["userId", "movieId", "rating"])

def test_init_cutoff_filtering():
    """Test that movies with fewer than min_ratings are filtered out."""
    cf = CollaborativeFilter(TEST_RATINGS, min_ratings=2)
    assert list(cf._movie_vectors.index) == [10, 20], "Movies 10 and 20 have 2 ratings each; 30 has only 1."

def test_init_matrix_shape():
    """Test that the movie-user matrix has the correct shape and orientation."""
    cf = CollaborativeFilter(TEST_RATINGS, min_ratings=1)
    assert list(cf._movie_vectors.index) == [10, 20, 30], "Rows should be movies."
    assert list(cf._movie_vectors.columns) == [1, 2], "Columns should be users."

def test_init_normalization():
    """Test that the movie vectors are normalized to have magnitude 1."""
    cf = CollaborativeFilter(TEST_RATINGS, min_ratings=1)
    norms = np.linalg.norm(cf._movie_vectors.to_numpy(), axis=1)
    assert norms == pytest.approx(1), "All movie vectors should have magnitude 1."

def test_init_movie_vectors_are_correct():
    """Test that the movie vectors are correctly normalized."""
    cf = CollaborativeFilter(TEST_RATINGS, min_ratings=1)
    norm = np.sqrt(5**2 + 3**2)
    assert cf._movie_vectors.loc[10, 1] == pytest.approx(5 / norm), "Movie 10's entry for user 1 is incorrect."
    assert cf._movie_vectors.loc[10, 2] == pytest.approx(3 / norm), "Movie 10's entry for user 2 is incorrect."

def test_similarity_to_itself_is_1(): 
    """Test that the similarity of a movie to itself is 1."""
    cf = CollaborativeFilter(TEST_RATINGS, min_ratings=1)
    similarities = cf._similarities([10])
    assert similarities.loc[10, 10] == pytest.approx(1), "A movie's similarity to itself should be 1."

def test_similarities_are_between_0_and_1():
    """Test that similarity scores are between 0 and 1."""
    cf = CollaborativeFilter(TEST_RATINGS, min_ratings=1)
    similarities = cf._similarities([10, 20])
    assert (similarities.to_numpy() >= 0).all() and (similarities.to_numpy() <= 1 + 1e-9).all(), "All similarity scores should be between 0 and 1."

def test_similarities_matches_hand_worked_example():
    """Test that the similarity scores match a hand-worked example."""
    cf = CollaborativeFilter(TEST_RATINGS, min_ratings=1)
    similarities = cf._similarities([10, 20])
    expected_similarity = (5*3 + 3*5) / (np.sqrt(5**2 + 3**2) * np.sqrt(3**2 + 5**2))
    assert similarities.loc[10, 20] == pytest.approx(expected_similarity), "Movies 10 and 20 should have similarity 30/34."

def test_similarities_movies_not_in_matrix_are_skipped():
    """Test that movies not in the matrix are skipped when computing similarities."""
    cf = CollaborativeFilter(TEST_RATINGS, min_ratings=1)
    similarities = cf._similarities([10, 20, 999])  
    assert list(cf._similarities([10, 20, 999]).index) == [10, 20], "Movie 999 should be skipped as it is not in the movie vectors."

def test_similarities_one_row_per_requested_movie():
    """Test that the resulting DataFrame has one row per requested movie and one column per movie in the matrix."""
    cf = CollaborativeFilter(TEST_RATINGS, min_ratings=1)
    similarities = cf._similarities([10, 20])
    assert list(similarities.index) == [10, 20], "The resulting DataFrame should have one row per requested movie."
    assert list(similarities.columns) == [10, 20, 30], "The resulting DataFrame should have one column per movie in the matrix." 

def test_breakdown_matches_hand_worked_example(): ...
def test_breakdown_weights_add_up_to_1(): ...
def test_breakdown_is_sorted_largest_first(): ...
def test_breakdown_leaves_out_movies_with_no_similarity(): ...