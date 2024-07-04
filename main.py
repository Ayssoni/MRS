import numpy as np
import pandas as pd  # CONTENT BASED RECOMMENDATION SYSTEM.

# desired_width=320
# pd.set_option('display.width', desired_width)
# pd.set_option('display.max_columns',23)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 23)
# import the datasets
movies = pd.read_csv('D:\Recommender_System\TMDB dataset\tmdb_5000_movies.csv')
credits = pd.read_csv('D:\Recommender_System\TMDB dataset\tmdb_5000_credits.csv')
# print(movies.head(1))
# print(credits.head(1).values)
# print(credits.head(1)['crew'].values)
# so here we have two dataframe of single project lets merge them, merge on movie id or name column
movies.merge(credits, on='title')
# print(movies.merge(credits,on='title').shape)
# print(movies.shape)
# print(credits.shape)
movies = movies.merge(credits, on='title')
# print(movies.head(1))
# we have 23 columns but in that we don't require all of them so lets divide them
# lets separate helping columns from the dataset
# print(movies.info())
# genres,id,keywords,title,overview,(release date),cast,crew.
# print(movies['original_language'].value_counts())
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
# print(movies.head(1))
# creating a tags column so in that i will place overview, geners, keywords,cast&crew
# in cast main three characters, crew - directors. and data preprocessing
# checking for is there any null values
# print(movies.isnull().sum())
# in overview we have 3 missing values so lets remove/ drop them
movies.dropna(inplace=True)
# print(movies.isnull().sum())
# lets check for duplicates in the dataset
# print(movies.duplicated().sum()) # no rwo duplicates are present
# now lets make the columns in correct formate
# print(movies.iloc[0].genres)
'''.iloc is a pandas DataFrame function used to select 
 elements in a DataFrame by integer position, as opposed to by label.
The .iloc function is used for integer-location based indexing / selection by position.'''
# [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
# we need this to be in this foramte ['Action','Adventure','Fantasy','ScienceFiction']
# print(ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'))
import ast
def convert(obj):
    L = []  # ast.literal_eval function is used to convert the input string into a Python list,
    for i in ast.literal_eval(obj):  # obj will be now in list.
        L.append(i['name'])
    return L

# print(movies['genres'].apply(convert))
# here we have genres column like this [Action, Adventure, Fantasy, Science Fiction].
movies['genres'] = movies['genres'].apply(convert)
movies["keywords"] = movies['keywords'].apply(convert)


# print(movies.head())
# print(movies['cast'][0]) # in this we need first  3 dict so in that we will get the main characters name
def convert3(obj):
    L = []  # ast.literal_eval function is used to convert the input string into a Python list,
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:  # obj will be now in list.
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(convert3)


# print(movies['cast'])
# for now crew
# print(movies['crew'][0]) # job = Director
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)
# print(movies['crew'])
# print(movies.head())
# print(movies['overview'][0])
# OVERVIEW is a Sring so now i need to convert it into list
movies['overview'] = movies['overview'].apply(lambda x: x.split())
# print(movies.head)
# here all we have is in list formate, i need to add all of them and
# convert them into string so that i end with a paragraph
# NOw here there is a problem we have in spaces at names like Science Fiction in this
# scienc and fiction are different tAGS SO WE NEEDE TO REMOVE THE SPACES . TRANSFORMATION
# movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
# print(movies['genres'][0])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
# print(movies.head())
# lets merge overview genres,keywords,cast and crew in a new column name tags
movies['tags'] = movies['overview'] + movies['genres'] + movies["keywords"] + movies['cast'] + movies['crew']
# print(movies.head()) #so here we can drop genres,keywords,cast and crew columns
Movies_Newdf = movies[['movie_id', 'title', 'tags']]
# print(Movies_Newdf.head()) # convert tags into string which is in list
# Movies_Newdf['tags']= Movies_Newdf['tags'].apply(lambda x:" ".join(x))
Movies_Newdf.loc[:, 'tags'] = Movies_Newdf['tags'].apply(lambda x: " ".join(x))
# print(Movies_Newdf.head())
# joins the elements of the list x into a single string,
# with each element separated by a space.
# print(Movies_Newdf['tags'][0]) # lets convert tags into lowercase
Movies_Newdf.loc[:, 'tags'] = Movies_Newdf['tags'].apply(lambda x: x.lower())
# print(Movies_Newdf['tags'][0])
#print(Movies_Newdf)
# print(Movies_Newdf['tags'][0]) # the goal is to create website in that if we give a movie name it should recommende 5 movies related to that.
import \
    nltk  #HERE IMPORT STATEMENT # here we need to do steaming for this we to import a lib called nltk , famous for NLP
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

#lets create a function
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


#print(ps.stem('loving'))
#print(stem('dancing'))
Movies_Newdf.loc[:, 'tags'] = Movies_Newdf['tags'].apply(stem)
#print(Movies_Newdf['tags'])
#print(Movies_Newdf['tags'][1]) # we need to check for the similarites in the tags
# as the tags are in string it is difficult, vectorization
from sklearn.feature_extraction.text import CountVectorizer  # HERE IMPORT STATEMENT

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(Movies_Newdf['tags']).toarray()
# print(vectors)
# print(vectors.shape)
# print(vectors[0])
fn = cv.get_feature_names_out()
#print(len(fn))
#print(fn) # in the vectors there are similar words like actor actors in place of 2 words we only use on actor
from sklearn.metrics.pairwise import cosine_similarity  #IMPORT
similarity = cosine_similarity(vectors)
#print(cosine_similarity(vectors)) #calculates distance between each vectors
#print(cosine_similarity(vectors).shape) # we are using cosine distance
#print(similarity[0].shape)# this is the distance between 4086 movies from 1st movie .more distance less similarity
#print(similarity[0])
#print(sorted(similarity[0],reverse=True)) # from this we need 1st 5movies in this index position is changed so we need to change code
#print(list(enumerate(similarity[0])))# from this we get index and distance for 1 movie with all movies
#we need to sort this now
#print(sorted(list(enumerate(similarity[0])),reverse=True)) # here we get sorted but on index based but we dont needd that we want on similarity based
#print(sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1]))# we need only 5 movies
#movies_list=sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
#fetch the index
#print(Movies_Newdf[Movies_Newdf['title']=='Batman Begins'].index[0])
def recommend(movie):  #here from similarity we will get the distances of movies  , we need to sort them in order most similar to less and produce 5 movies
    movie_index = Movies_Newdf[Movies_Newdf['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(Movies_Newdf.iloc[i[0]].title)

#print(recommend('Batman Begins')) # here we get index numbers but we need names
#print(Movies_Newdf.iloc[1214].title)
#print(recommend('Avatar'))

# so now i cant search a new movie because it was not in the dataset &
# i need to a add a new movie to the dataset on the movie id, movie name , tags
# take these as a input and from this i can recommend new movies also
