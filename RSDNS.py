import numpy as np
import pandas as pd
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 25)

# Read data from CSV files
RawMovies = pd.read_csv("/Users/bharathgoud/PycharmProjects/MovieRecomndationSystem/Recommender_System/TMDB dataset/tmdb_5000_movies.csv")
RawCredits = pd.read_csv("/Users/bharathgoud/PycharmProjects/MovieRecomndationSystem/Recommender_System/TMDB dataset/tmdb_5000_credits.csv")

# Merge the dataframes on 'title' column
MergeDF = pd.merge(RawMovies, RawCredits, on='title')

# Select required columns
movies = MergeDF[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Select required columns
RequiredColumnDF = MergeDF[['movie_id', 'title', 'overview', 'keywords', 'genres', 'cast', 'crew']]

# Check for null values and remove rows with null 'overview'
null_values = RequiredColumnDF.isnull().sum()
RequiredColumnDF = RequiredColumnDF.dropna(subset=['overview']).copy()
null_values_after = RequiredColumnDF.isnull().sum()

# Remove duplicates
duplicates = RequiredColumnDF.duplicated().sum()

# Convert string representation of lists to actual lists
import ast
def convert(input_string):
    updated = []
    for i in ast.literal_eval(input_string):
        updated.append(i['name'])
    return updated

RequiredColumnDF['genres'] = RequiredColumnDF['genres'].apply(convert)
RequiredColumnDF["keywords"] = RequiredColumnDF["keywords"].apply(convert)

def convert2(input_string_cast):
    updated = []
    counter = 0
    for i in ast.literal_eval(input_string_cast):
        if counter != 3:
            updated.append(i['name'])
            counter += 1
        else:
            break
    return updated

RequiredColumnDF['cast'] = RequiredColumnDF['cast'].apply(convert2)

def fetch_directorName(input_string_crew):
    Dname = []
    for i in ast.literal_eval(input_string_crew):
        if i['job'] == 'Director':
            Dname.append(i['name'])
            break
    return Dname

RequiredColumnDF['crew'] = RequiredColumnDF['crew'].apply(fetch_directorName)

# Tokenize 'overview' column
RequiredColumnDF['overview'] = RequiredColumnDF['overview'].apply(lambda x: x.split())

# Remove spaces from individual items in lists
RequiredColumnDF['genres'] = RequiredColumnDF['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
RequiredColumnDF['keywords'] = RequiredColumnDF['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
RequiredColumnDF['cast'] = RequiredColumnDF['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
RequiredColumnDF['crew'] = RequiredColumnDF['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
RequiredColumnDF['overview'] = RequiredColumnDF['overview'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create 'tags' column by combining other columns
RequiredColumnDF['tags'] = RequiredColumnDF['genres'] + RequiredColumnDF['cast'] + RequiredColumnDF['crew'] + RequiredColumnDF['keywords'] + RequiredColumnDF['overview']

# Select relevant columns for MoviesDF
MoviesDF = RequiredColumnDF[['movie_id', 'title', 'tags']]

# Join tags into a single string and preprocess
MoviesDF.loc[:, 'tags'] = MoviesDF['tags'].apply(lambda x: " ".join(x))
MoviesDF.loc[:, 'tags'] = MoviesDF['tags'].apply(lambda x: x.lower())

# Stemming
from nltk.stem.porter import PorterStemmer
Ps = PorterStemmer()
def stem(input_text):
    basewords = []
    for i in input_text.split():
        basewords.append(Ps.stem(i))
    return " ".join(basewords)

MoviesDF.loc[:, 'tags'] = MoviesDF['tags'].apply(stem)

# Create CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(MoviesDF['tags']).toarray()
fn = cv.get_feature_names_out()

# Compute similarity matrix
similarity = cosine_similarity(vectors)

import pickle

def AddMovie(movie_id, title, tags):
    global MoviesDF, vectors, similarity
    new_movie = {'movie_id': [movie_id], 'title': [title], 'tags': [tags]}
    new_movie_df = pd.DataFrame(new_movie)
    MoviesDF = pd.concat([MoviesDF, new_movie_df], ignore_index=True)

    # Apply preprocessing steps to the new movie data
    MoviesDF.loc[len(MoviesDF) - 1, 'tags'] = MoviesDF.loc[len(MoviesDF) - 1, 'tags'].lower()
    MoviesDF.loc[len(MoviesDF) - 1, 'tags'] = PorterStemmer().stem(MoviesDF.loc[len(MoviesDF) - 1, 'tags'])

    # Update vectors and similarity matrices
    vectors = cv.transform(MoviesDF['tags']).toarray()
    similarity = cosine_similarity(vectors)

    # Save updated MoviesDF dataset to pickle file
    with open('MoviesDF.pickle', 'wb') as f:
        pickle.dump(MoviesDF, f)
def RecommendMovies(movie):
    global MoviesDF, similarity
    movie_indices = MoviesDF[MoviesDF['title'] == movie].index
    if len(movie_indices) == 0:
        print("Sorry, the movie '{}' is not in the dataset.".format(movie))
        # Take input for movie ID, title, and tags
        movie_id = int(input("Enter the movie ID: "))
        title = input("Enter title of the movie: ")
        tags = input("Enter the tags for the movie (separated by commas): ")

        # Add the new movie to the dataset
        AddMovie(movie_id, title, tags)

        # Get recommendations for the new movie
        movie_index = len(MoviesDF) - 1
        distance = similarity[movie_index]
        movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        print("Recommended movies for '{}' (Newly added):".format(title))
        for i in movies_list:
            print(MoviesDF.iloc[i[0]].title)
    else:
        movie_index = movie_indices[0]
        distance = similarity[movie_index]
        movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        print("Recommended movies for '{}':".format(movie))
        for i in movies_list:
            print(MoviesDF.iloc[i[0]].title)

RecommendMovies('Salaar')