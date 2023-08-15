#author : Sohaib Khan
#topic : Bagging and its proces 

from random import seed
from random import randrange
from csv import reader

# Loading file
def loadfile(filename=("Patient data.csv")):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Changing a string into a floating point number
def stringtofloat(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Transform a string column to an integer
def stringtoint(dataset, column):
	classval = [row[column] for row in dataset]
	unique = set(classval)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Create k folds from a dataset.
def crossval_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Apply cross validation split to the evaluation of an algorithm.
def eval_algo(dataset, algorithm, n_folds, *args):
	folds = crossval_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Split a dataset based on an attribute and an attribute value
def testsplit(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classval):
	gini = 0.0
	for class_value in classval:
		for group in groups:
			size = len(group)
			if size == 0:
				continue
			proportion = [row[-1] for row in group].count(class_value) / float(size)
			gini += (proportion * (1.0 - proportion))
	return gini

# A dataset's ideal split point is chosen.
def get_split(dataset):
	classval = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
		# for i in range(len(dataset)):
		# 	row = dataset[randrange(len(dataset))]
			groups = testsplit(index, row[index], dataset)
			gini = gini_index(groups, classval)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def toterminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, maximum_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = toterminal(left + right)
		return
	# check for maximum depth
	if depth >= maximum_depth:
		node['left'], node['right'] = toterminal(left), toterminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = toterminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], maximum_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = toterminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], maximum_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, maximum_depth, min_size):
	root = get_split(train)
	split(root, maximum_depth, min_size, 1)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Make a prediction with a list of bagged trees
def predictionbagging(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

# Bootstrap Aggregation Algorithm
def bagging(train, test, maximum_depth, min_size, sample_size, n_trees):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, maximum_depth, min_size)
		trees.append(tree)
	predictions = [predictionbagging(trees, row) for row in test]
	return(predictions)

# Test bagging on the sonar dataset
seed(1)
# load and prepare data
filename = 'Patient data.csv'
dataset = loadfile(filename)
# convert string attributes to integers
for i in range(len(dataset[0])-1):
	stringtofloat(dataset, i)
# convert class column to integers
stringtoint(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 6
maximum_depth = 7
min_size = 3
sample_size = 0.65


for n_trees in [2, 4, 8, 40]:
	scores = eval_algo(dataset, bagging, n_folds, maximum_depth, min_size, sample_size, n_trees)
	print('Trees are: %d' % n_trees)
	print('Scores are: %s' % scores)
	print('Mean Accuracy is: %.3f%%' % (sum(scores)/float(len(scores))))