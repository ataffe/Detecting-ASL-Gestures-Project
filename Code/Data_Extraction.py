import csv
import numpy as np



"""
function 
	The purpose of the funtion is to read from a csv file of a known format and produce the numerical data, feature names, and labels

inputs
	filename - the name of a csv file to be open and read

outputs
	data - The numerical data from the file.
	featureLabels - The name given to each column of the file indicating the feature names
	dataLabels - The labels indicating what letter the information in a row belongs to

analysis
	Let M denote the number of rows, and N denote the number of columns in the CSV file
	Lines 1 - 4 take constant time: O(1)
	Lines 5 - 6 times are proportinal to the number of rows: O(M)
	Lines 7 copies a 2D array (matrix) of data: O(MN)
	Lines 8 copies the first row of the data: O(N)
	Lines 9 copies the first column of the data: O(M)
	Lines 10 deletes the first row: O(MN)
	Lines 11 deletes the first column O(MN)
	Lines 12 converts all data: O(MN)
	Lines 13 - 14 converts labels into the single letter: O(N)

	aM + bMN + cN + dM + eMN + fMN + gMN + hN
	= (b + e + f + g)MN + (c + h0)N + (a + d)M
	= O(MN)
	The time is proportional to the number of features times the number letter data gathered
"""
def extractData(filename):
	LABEL_INDEX = 9									# A constant value used to format the data labels

	file = open(filename, 'r')						# Open the file for reading
	reader = csv.reader(file)						# Create CSV file reading object

	rawData = []									# Put the CSV file into a list
	for row in reader:
		rawData = rawData + [row]

	data = np.array(rawData)						# Turn list of data into a numpy array

	featureLabels = data[0, 1:]						# Pull the feature catagories from the first row of the array
	dataLabels = data[1:, 0]						# Pull the labels from the first column of the array

	data = np.delete(data, 0, axis = 0)				# Remove the feature categories from the data
	data = np.delete(data, 0, axis = 1)				# Remove the labels from the data
	data = data.astype(np.float)					# Convert data from strings to floats

	# Format the labels into a usable form
	for index in range(dataLabels.size):
		dataLabels[index] = dataLabels[index][LABEL_INDEX]

	return (data, featureLabels, dataLabels)



"""
function
	The purpose of the function is to take the data before and after feature selection and determine which features were selected

inputs
	data - The numerical data from the file.
	featureLabels - The name given to each column of the file indicating the feature names
	dataLabels - The labels indicating what letter the information in a row belongs to

outputs
	newFeatures - An array of the feature names chosen by feature selection

analysis
	Lines 1 - 2 take constant time: O(1)
	Lines 3 - 6 iterate through the length of a row in data: O(N)

	O(N)
	The time is proportional to the number of features
"""
def extractFeatures(data, newData, featureLabels):
	newDataIndex = 0													# A "pointer" for newData
	newFeatures = []

	for dataIndex in range(data[0].size):
		if data[0][dataIndex] == newData[0][newDataIndex]:				# If value at index dataIndex in data[0] array equals value at index newDataIndex in newData[0] array
			newFeatures = newFeatures + [featureLabels[dataIndex]]		# If a match is found, take note of what feature was selected by SelectKBest
			newDataIndex += 1	

	return newFeatures



"""
function
	The function takes in the data and its labels, and splits the data into two sets: training and testing

inputs
	data - The numerical data extracted from the CSV file
	dataLabels - The dataLabels extracted from the CSV file that associate with the data

outputs
	trainingData - The numerical data which will be used for training
	testData - The numberical data which will be used for testing
	trainingLabels - TrainingData's associated labels
	testLabels - testingData's associated labels

analysis
	Lines 1 - 6 take constant time: O(1)
	Lines 6 - 20 iterate through the length of dataLabels: O(M)

	O(M)
	The time is propostional to the amount of gathered labels which need processing
"""
def splitData(data, dataLabels):
	count = 0				# A counter for helping to determine the split of data into test and training sets
	label = None			# The currect label being exaimined

	testData = []			# A list to hold the data used for testing
	testLabels = []			# A list to contain the labels which accompany the test data

	trainingData = []		# A list to hold the data used for training
	trainingLabels = []		# A list to contain the labels which accompany the training data

	# Iterate through all the labels of the input file and split them into two groups: training and testing
	for index in range(dataLabels.size):
		if label == dataLabels[index]:										# If the previous label is the same as the current label
			if count < 3:													# And 3 of this label havn't been seen
				trainingData = trainingData + [data[index]]					# Mark this label and its associated data for training
				trainingLabels = trainingLabels + [dataLabels[index]]
				count += 1
			else:															# Else, if we've seen more than 3, mark the data and label for testing
				testData = testData + [data[index]]
				testLabels = testLabels + [dataLabels[index]]
		else:																# If the label previous label is different from the current label
			trainingData = trainingData + [data[index]]						# Mark it for training
			trainingLabels = trainingLabels + [dataLabels[index]]
			label = dataLabels[index]
			count = 1

	return (trainingData, testData, trainingLabels, testLabels)



"""
function
	The function removes the data no longer needed in testData. The data no longer needed was determined by feature selection

inputs
	testData - The data split from the training data intend to test the model
	features - The entire set of features produced from normaliztion
	selectedFeatures - The features selected from the feature selection process

outputs
	newTestData - The reduced set of testData which should actually be used for testing

analysis
	Lines 1 - 2 take constant time: O(1)
	Lines 3 - 7: O(N + k): Although the while loop is nested, it takes advantage of ordering and only actually iterates the length of features once
		Lines 3 iterates through the selectedFeatures once, which is of size k: O(k)
		Lines 4 - 5 will iterate through features once, in a stop and go fashion: O(N)
		Lines 6 - 7 will execure every for-loop iteration: O(k)
	Lines 8: O(1)
	Lines 9 - 13: The outer for-loop iterates O(M) times, and the inner for-loop iterates O(k) times: O(Mk)

	= O(N + k) + O(Mk) + O(N) + O(k) + O(k)

	O(Mk)
	The time is propostional to the original amount of features times the reduced amount of features
"""
def reduceData(testData, features, selectedFeatures):
	indecies = []				# Holds the indecies for the selected features
	featureIndex = 0			# A placeholder to take advantage of ordering

	for selFeatureIndex in range(len(selectedFeatures)):															# Iterate through all selectedFeatures
		while (featureIndex < len(features)) and (features[featureIndex] != selectedFeatures[selFeatureIndex]):		# Iterate through features until finding the selectedFeature
			featureIndex += 1

		if features[featureIndex] == selectedFeatures[selFeatureIndex]:			# If the selectedFeature is found, note its index (the column it belongs to)
			indecies = indecies + [featureIndex]



	newTestData = []						# newTetData will hold the reduced data set

	for row in testData:					# Iterate through all rows in testData
		temp = []							# Will hold the appropriate values from the row
		for index in indecies:				# Once appropriate feature value is found, add it to temp
			temp = temp + [row[index]]

		newTestData = newTestData + [temp]	# Add temp as a row to the new data

	return newTestData