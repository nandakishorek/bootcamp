from flask import Flask
from flask import jsonify
from flask import request
import pandas as pd
import numpy as np
import random
import csv

app = Flask('bootcamp')

ratings_list = [i.strip().split(",") for i in open('../data/ratings.csv', 'r').readlines()]

with open('../data/movies.csv') as csvfile:
    reader = csv.reader(csvfile)
    movies_list = [row for row in reader]

ratings_list = ratings_list[1:];
movies_list = movies_list[1:];

def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
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

	user_id = json_body['user_id']
	ratings = json_body['']

	ratings = np.array(ratings_list)
	movies = np.array(movies_list)

	ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
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

	all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

	preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
	preds_df.head()

def handle_get():
	idx = len(ratings_list) - 1
	user_id = int(ratings_list[idx][0]) + 1
	movies = random.sample(movies_list, 6)
	movie_list = [{'id':movie[0], 'name':movie[1], 'genres':movie[2].split('|')} for movie in movies]
	return jsonify({'user_id':user_id,'movies':movie_list})

@app.route('/recommendation', methods=['POST', 'GET'])
def recommendation():
	if (request.method == 'POST'):
		return handle_post(request.json)
	elif (request.method == 'GET'):
		return handle_get()
	
