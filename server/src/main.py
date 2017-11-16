from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
import pandas as pd
import numpy as np
import random
import csv
import time

app = Flask('bootcamp')
CORS(app)

ratings_list = [i.strip().split(",") for i in open('../data/ratings.csv', 'r').readlines()]

with open('../data/movies.csv') as csvfile:
    reader = csv.reader(csvfile)
    movies_list = [row for row in reader]

ratings_list = ratings_list[1:];
movies_list = movies_list[1:];

def recommend_movies(preds_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) # UserID starts at 1
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False)
                 )

    # print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
    # print 'Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations)
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'MovieID',
               right_on = 'MovieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :]
                      )
    
    return user_full, recommendations

def handle_post(json_body):
	print('handle_post')
	user_id = json_body['user_id']
	ip_movies = json_body['movies']

	for movie in ip_movies:
		ratings_list.append([user_id,movie['id'],movie['rating'],time.time()])

	print('appended rating')
	ratings = np.array(ratings_list)
	movies = np.array(movies_list)

	ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
	#ratings_df.drop_duplicates(keep=False)

	movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
	movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

	movies_df.head()
	ratings_df.head()

	R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
	R_df.head()
	R1 = R_df.as_matrix()
	R = [[float(y) for y in x] for x in R1]
	user_ratings_mean = np.mean(R, axis = 1)
	R_demeaned = R - user_ratings_mean.reshape(-1, 1)

	from scipy.sparse.linalg import svds
	U, sigma, Vt = svds(R_demeaned, k = 50)

	sigma = np.diag(sigma)

	print('predicting...')

	all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

	preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
	preds_df.head()

	already_rated, predictions = recommend_movies(preds_df, int(user_id), movies_df, ratings_df, 10)

	print('predicted ' + str(predictions))
	return predictions.to_json()

def handle_get():
	idx = len(ratings_list) - 1
	user_id = int(ratings_list[idx][0]) + 1
	movies = random.sample(movies_list, 6)
	movie_list = [{'id':movie[0], 'name':movie[1], 'genres':movie[2].split('|')} for movie in movies]
	return jsonify({'user_id':user_id,'movies':movie_list})

@app.route('/recommendation', methods=['GET','POST'])
def recommendation():
	print('recommending...')
	if (request.method == 'POST'):
		print('handling post method...')
		return handle_post(request.json)
	elif (request.method == 'GET'):
		print('handling get method...')
		return handle_get()
	
