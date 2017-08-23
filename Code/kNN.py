import csv
import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier

import Data_Extraction



K_SELECT_MIN = 15
K_SELECT = 511
K_SELECT_MAX = 510
STEP_SIZE = 5

K_NEIGHBORS_MIN = 1
K_NEIGHBORS = 11



data, features, labels = Data_Extraction.extractData('features_normalized.csv')					# Extract the data, features, and labels from the CSV file

trainingData, testData, trainingLabels, testLabels = Data_Extraction.splitData(data, labels)	# Split the data into a set for training and a set for testing

neighbors = range(K_NEIGHBORS_MIN, K_NEIGHBORS)			# Create the list of neighbors to select during test
selectBest = range(K_SELECT_MIN, K_SELECT, STEP_SIZE)	# Create the list of features to select during test

file = open('scores.csv', 'wb')										# Create a file called scores.csv
writer = csv.writer(file, dialect = 'excel')						# Create a CSV file writer
writer.writerow(["k_neighbors", "k_features", "score", "time"])		# Writer the columns tags



for k_neighbors in neighbors:
	for k_features in selectBest:
		startTime = time.clock()

		trainedData = SelectKBest(f_classif, k = k_features).fit_transform(trainingData, trainingLabels)	# Perform feature selection with SelectKBest
		selectedFeatures = Data_Extraction.extractFeatures(data, trainedData, features)						# Determine which features were chosen as the K best
		newTestData = Data_Extraction.reduceData(testData, features, selectedFeatures)						# Reduce the test data via results of feature selection						

		kNN = KNeighborsClassifier(n_neighbors = k_neighbors)	# Create a kNN classifer						
		kNN.fit(trainedData, trainingLabels)					# Fit the training data to the classifier
		score = kNN.score(newTestData, testLabels)				# Test the classifier

		endTime = time.clock()

		overallTime = endTime - startTime

		writer.writerow([k_neighbors, k_features, score, overallTime])		# Write results to CSV file for k-neighbors and select-k-best