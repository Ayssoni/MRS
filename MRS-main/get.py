import pickle
import pandas as pd

with open('MoviesDF.pickle', 'rb') as f:
    MoviesDF = pickle.load(f)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(MoviesDF['tags']).toarray()
similarity = cosine_similarity(vectors)

def get_recommendation(movie_name):

    movie_name = movie_name.strip()
    
    if movie_name not in MoviesDF['title'].values:
        return f"Error: '{movie_name}' not found in the dataset. Please check the spelling."

    movie_index = MoviesDF[MoviesDF['title'] == movie_name].index[0]
    
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    print(f"\nTop 5 movies similar to '{movie_name}':")
    for i in movies_list:
        print(f"- {MoviesDF.iloc[i[0]].title}")

# Type your movie here
my_choice = "kgf"
get_recommendation(my_choice)