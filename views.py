from flask import render_template
from flask import request
from flask_flute import app
from flask_flute.model_one import model_it
import pandas as pd
import numpy as np
import spotipy
import requests
import re
import json
import pickle
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

# initialize with path, spotipy id
path = 'yourPathHere' # where pickled files are stored

client_id = 'yourID' # for Spotify web API
client_secret = 'yourClientSecret'

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# define important features, number of nearest neighbors for model
n_neighbors = 5
elements = ["key_confidence","tempo_confidence","mode_confidence",
"time_signature_confidence","key_difficulty"]

def findOnSpotify(title, composer):
	"""Find piece on Spotify"""
	searchStr = title + ' ' + composer
	pieces_played_all = sp.search(searchStr)
	pieces_played = pieces_played_all['tracks']
	piece_played = pieces_played['items']
	if piece_played == []:
		id_piece = []
	else:
		# Pick first piece found
		piece_played_details = piece_played[0]
		id_piece = piece_played_details['id']
		track_piece = piece_played_details['name']
	return id_piece

def retrieveFeatures(track_id):
	"""Find track features through Spotify"""
	track_info = sp.audio_analysis(track_id)
	track_details = track_info['track']
	return track_details
	
def extractFeatures(track_details, features):
	feature_dict = {}	
	for feature in features:
		feature_dict[feature] = track_details[feature]
	return feature_dict

def unpickle(pickleFile):
	"""Unpickle saved data set"""
	pickle_off = open(pickleFile,"rb")
	unpickledFile = pickle.load(pickle_off)
	return unpickledFile

def id_major_key(key):
	"""Define difficulty of major keys"""
	if key == 0 or key == 10 or key == 5:
		key_diff = 0
	elif key == 3 or key == 2 or key == 7:
		key_diff = 0.33
	elif key == 8 or key == 4 or key == 9:
		key_diff = 0.67
	elif key == 1 or key == 6 or key == 11:
		key_diff = 1
	return key_diff

def id_minor_key(key):
	"""Define difficulty of minor keys"""
	if key == 9 or key == 2 or key == 7:
		key_diff = 0
	elif key == 0 or key == 11 or key == 4:
	    key_diff = 0.33
	elif key == 5 or key == 1 or key == 6:
	    key_diff = 0.67
	elif key == 10 or key == 3 or key == 8:
		key_diff = 1
	return key_diff

def id_key_diff(key,mode):
	"""Find key difficulty based on mode"""
	if mode == 1: # Major Key
		key_diff = id_major_key(key)
	else: # Minor Key
		key_diff = id_minor_key(key)
	return key_diff

def add_key_grade(df):
	"""find key difficulty of piece"""
	key_difficulty = []
	grades = []
	for row in df.iterrows():
		mode = row[1]["mode"]
		key = row[1]["key"]
		
		key_diff = id_key_diff(key,mode)

		key_difficulty.append(key_diff)
	return key_difficulty

def categorizeDist(nn_indices):
	"""Categorize distance of nearest neighbor as quality of match"""
	distance_pieces = []
	for index in nn_indices:
		if index < 0.15:
			distance_piece = 'Excellent'
		elif index < 0.25:
			distance_piece = 'Very Good'
		elif index < 0.35:
			distance_piece = 'Good'
		elif index < 0.5:
			distance_piece = 'Fair'		
		else:
			distance_piece = 'Poor'
		distance_piece = distance_piece #+ ', ' + str(round(index,2))
		distance_pieces.append(distance_piece)
	return distance_pieces

@app.route('/')
@app.route('/input')
def flute_input():
    return render_template("input.html")

@app.route('/output')
def flute_output():
	
	# Pull 'title', 'composer', 'suggestion number' from input field and store it
	title_input = request.args.get('title_played')
	composer_input = request.args.get('composer_played')
	suggestion_input = 3#request.args.get('rec_num')
	
	# Decide which database to access depending on flute music wanted
	request_input = request.args.get('specialReq')
	request_input = request_input.lower()

	if request_input == 'both':
		dbtable = unpickle(path+'df_pieces_all_2.pickle')
	elif request_input == 'flute and piano':
		dbtable = unpickle(path+'df_pieces_piano_2.pickle')
	elif request_input == 'solo flute':
		dbtable = unpickle(path+'df_pieces_solo_2.pickle')

	# Find spotify id of piece played on spotify
	id_piece = findOnSpotify(title_input, composer_input)
	try:
		piece_features = retrieveFeatures(id_piece)
		features = ['duration', 'tempo', 'tempo_confidence', 'time_signature',
			'time_signature_confidence', 'key', 'key_confidence', 'mode', 				'mode_confidence']
		feature_dict = extractFeatures(piece_features, features)
		
		mode = feature_dict["mode"]
		key = feature_dict["key"]
		feature_dict['key_difficulty'] = id_key_diff(key,mode)
	except:
		result = "We can't find what you're looking for...please try another piece or simplify your title/composer!"
		return result
	
	df = dbtable
	
	# Add column regarding key difficulty
	key_difficulty = add_key_grade(df)
	df['key_difficulty'] = key_difficulty

	# Create testing and training data sets
	x = df.as_matrix(elements)
	y = df["grade"]
	
	# round grades to whole numbers (ie 3+ -> 3, 3- -> 3)
	y_new = []
	for grade in y:
		grade = grade[0][0]
		y_new.append(grade)
	
	x_train, x_test, y_train, y_test = train_test_split		(x,y_new,test_size=0.2,random_state=12)

	# Use k-nearest neighbors to identify grade of new piece and potential matches
	neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
	neigh.fit(x_train,y_train)

	# Validation - how good is the algorithm?
	score = neigh.score(x_test,y_test)
	score = str(score)

	# What is the grade of the piece you just played?
	feat_list =[]
	for element in elements:
		feat_list.append(feature_dict[element])

	grade_played = neigh.predict(feat_list)

	# What are the nearest neighbors?
	nn_indices = neigh.kneighbors(feat_list,n_neighbors=int(suggestion_input))
	match_pieces = nn_indices[1][0]
	distance_pieces = nn_indices[0][0]
	distance_pieces = categorizeDist(nn_indices[0][0])

	titles = []
	composers = []
	i = 0
	
	for match in match_pieces:
		track_id = df.loc[match]['trackID']
		url = "https://open.spotify.com/embed?uri="+"spotify:track:"+track_id+"&theme=white"
		titles.append((df.loc[match]['title'], df.loc[match]['composer'], distance_pieces[i],url))
		i = i + 1

	spotify_track_url = "https://open.spotify.com/embed?uri="+"spotify:track:"+id_piece+"&theme=white"

	#return result	
	return render_template("output.html", titles=titles, spotify_track_url=spotify_track_url)

@app.route('/examples')
def flute_examples():
    return render_template("examples.html")

@app.route('/about')
def flute_about():
    return render_template("about.html")

@app.route('/contact')
def flute_contact():
    return render_template("contact.html")
		
