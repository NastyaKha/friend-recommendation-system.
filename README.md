#KHA ANASTASIIA

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
movie_ratings = pd.merge(ratings, movies, on='movieId')
# Creating a user rating matrix for each film
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating')
user_movie_ratings = user_movie_ratings.fillna(0)
movies['genres'] = movies['genres'].fillna('')
movies['genres'] = movies['genres'].apply(lambda x: x.replace('|', ' '))

#  Vectorization of genre texts
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

# Computing cosine similarity between movies based on their genres
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to predict a user's rating based on movies
def predict_user_rating(user_movie_ratings, cosine_sim):
    user_profile = user_movie_ratings.dot(cosine_sim)
    return user_profile

# Example of predicted ratings for a user
user_id = 1
user_ratings = predict_user_rating(user_movie_ratings.loc[user_id], cosine_sim)
user_ratings_df = pd.DataFrame(user_ratings, index=user_movie_ratings.columns, columns=['Relevance'])
user_ratings_df = user_ratings_df.sort_values(by='Relevance', ascending=False)

# Displaying top-10 recommendations with relevance
top_recommendations = user_ratings_df.head(10)
print(top_recommendations)
