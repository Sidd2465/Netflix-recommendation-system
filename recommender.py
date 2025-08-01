import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_similarity_matrix(df):
    pivot = df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    sim_matrix = cosine_similarity(pivot)
    sim_df = pd.DataFrame(sim_matrix, index=pivot.index, columns=pivot.index)
    return sim_df

def recommend_movies(movie_name, similarity_df, top_n=5):
    if movie_name not in similarity_df.columns:
        return ["Movie not found in database."]
    sim_scores = similarity_df[movie_name].sort_values(ascending=False)[1:top_n+1]
    return sim_scores.index.tolist()
