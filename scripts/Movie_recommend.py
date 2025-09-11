import pandas as pd
import numpy as np

# ------------------------------
# Example data
# ------------------------------

users = pd.DataFrame([
    {"user_id": 1, "full_name": "Ryan James", "age": 24, "gender": "M", "zip_code": "85711"},
    {"user_id": 2, "full_name": "Alice Johnson", "age": 30, "gender": "F", "zip_code": "94043"},
    {"user_id": 3, "full_name": "Bob Smith", "age": 23, "gender": "M", "zip_code": "32067"},
])

movies = pd.DataFrame([
    {"movie_id": 101, "movie_title": "Alpha", "release_year": 2020},
    {"movie_id": 102, "movie_title": "Beta", "release_year": 2020},
    {"movie_id": 103, "movie_title": "Gamma", "release_year": 2020},
    {"movie_id": 104, "movie_title": "Delta", "release_year": 2020},
    {"movie_id": 105, "movie_title": "Omega", "release_year": 2021},
])

ratings = pd.DataFrame([
    {"user_id": 1, "item_id": 101, "rating": 5, "timestamp": 1000},
    {"user_id": 1, "item_id": 102, "rating": 3, "timestamp": 1010},
    {"user_id": 2, "item_id": 101, "rating": 4, "timestamp": 1005},
    {"user_id": 2, "item_id": 103, "rating": 5, "timestamp": 1020},
    {"user_id": 3, "item_id": 101, "rating": 5, "timestamp": 1030},
    {"user_id": 3, "item_id": 104, "rating": 4, "timestamp": 1040},
])
ratings = ratings.astype({"user_id": int, "item_id": int, "rating": float, "timestamp": int})

# ------------------------------
# Helper: check similarity
# ------------------------------

def check_similar_users(ratings_df: pd.DataFrame, user_x: int, user_y: int) -> bool:
    """Return True if two users are similar (per problem spec)."""
    rx = ratings_df[ratings_df["user_id"] == user_x][["item_id", "rating"]].rename(columns={"rating": "r_x"})
    ry = ratings_df[ratings_df["user_id"] == user_y][["item_id", "rating"]].rename(columns={"rating": "r_y"})
    merged = pd.merge(rx, ry, on="item_id", how="inner")
    if merged.empty:
        return False
    merged["abs_diff"] = (merged["r_x"] - merged["r_y"]).abs()
    return merged["abs_diff"].max() <= 1.0

# ------------------------------
# Main: recommendation function
# ------------------------------

def get_recommendations(users_df, movies_df, ratings_df, full_name, year, method) -> str:
    """Return one recommended movie title for the given user, year, and method."""
    
    valid_methods = {"by_popularity", "by_rating", "by_similar_users"}
    if method not in valid_methods:
        return ""

    # Find the target user
    user_row = users_df[users_df["full_name"] == full_name]
    if user_row.empty:
        return ""
    user_id = int(user_row.iloc[0]["user_id"])

    # Filter movies by year
    if year not in movies_df["release_year"].values:
        return ""
    movies_year = movies_df[movies_df["release_year"] == year].copy()
    if movies_year.empty:
        return ""

    # Candidate movies: exclude ones already rated by user
    user_rated = set(ratings_df[ratings_df["user_id"] == user_id]["item_id"].unique())
    candidates = movies_year[~movies_year["movie_id"].isin(user_rated)].copy()
    if candidates.empty:
        return ""

    # --- Method 1: by_popularity ---
    if method == "by_popularity":
        counts = (
            ratings_df[ratings_df["item_id"].isin(candidates["movie_id"])]
            .groupby("item_id", as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        merged = candidates.merge(counts, left_on="movie_id", right_on="item_id", how="left").fillna({"count": 0})
        max_count = merged["count"].max()
        top = merged[merged["count"] == max_count]
        top_sorted = top.sort_values("movie_title")
        return str(top_sorted.iloc[0]["movie_title"])

    # --- Method 2: by_rating ---
    if method == "by_rating":
        means = (
            ratings_df[ratings_df["item_id"].isin(candidates["movie_id"])]
            .groupby("item_id", as_index=False)["rating"]
            .mean()
            .rename(columns={"rating": "avg_rating"})
        )
        merged = candidates.merge(means, left_on="movie_id", right_on="item_id", how="left")
        merged = merged.dropna(subset=["avg_rating"])
        if merged.empty:
            return ""
        max_avg = merged["avg_rating"].max()
        top = merged[merged["avg_rating"] == max_avg]
        top_sorted = top.sort_values("movie_title")
        return str(top_sorted.iloc[0]["movie_title"])

    # --- Method 3: by_similar_users ---
    if method == "by_similar_users":
        other_users = users_df[users_df["user_id"] != user_id]["user_id"].unique()
        similar = [int(u) for u in other_users if check_similar_users(ratings_df, user_id, int(u))]
        if len(similar) == 0:
            return ""
        sim_ratings = ratings_df[
            (ratings_df["user_id"].isin(similar)) & (ratings_df["item_id"].isin(candidates["movie_id"]))
        ].copy()
        if sim_ratings.empty:
            return ""
        latest = (
            sim_ratings.groupby("item_id", as_index=False)["timestamp"]
            .max()
            .rename(columns={"timestamp": "latest_ts"})
        )
        merged = candidates.merge(latest, left_on="movie_id", right_on="item_id", how="inner")
        max_ts = merged["latest_ts"].max()
        top = merged[merged["latest_ts"] == max_ts]
        top_sorted = top.sort_values("movie_title")
        return str(top_sorted.iloc[0]["movie_title"])

    return ""  # fallback

# ------------------------------
# Example runs
# ------------------------------

print(get_recommendations(users, movies, ratings, "Alice Johnson", 2020, "by_popularity"))
print(get_recommendations(users, movies, ratings, "Alice Johnson", 2020, "by_rating"))
print(get_recommendations(users, movies, ratings, "Alice Johnson", 2020, "by_similar_users"))
