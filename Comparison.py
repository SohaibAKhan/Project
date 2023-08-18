# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 20:28:03 2023

@author: skrox
"""

from random import seed
from random import randrange
from csv import reader
from math import sqrt

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

def bag_gini_index(groups, classval):
	bag_gini = 0.0
	for class_value in classval:
		for group in groups:
			size = len(group)
			if size == 0:
				continue
			proportion = [row[-1] for row in group].count(class_value) / float(size)
			bag_gini += (proportion * (1.0 - proportion))
	return bag_gini

def rf_gini_index(groups, classes):
    # count all samples at split point
 n_instances = float(sum([len(group) for group in groups]))
 # sum weighted Gini index for each group
 rf_gini = 0.0
 for group in groups:
     size = float(len(group))
 # avoid divide by zero
     if size == 0:
         continue
 score = 0.0
 # score the group based on the score for each class
 for class_val in classes:
     p = [row[-1] for row in group].count(class_val) / size
     score += p * p
 # weight the group score by its relative size
 rf_gini += (1.0 - score) * (size / n_instances)
 return rf_gini

def bag_get_split(dataset):
	classval = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
		# for i in range(len(dataset)):
		# 	row = dataset[randrange(len(dataset))]
			groups = testsplit(index, row[index], dataset)
			bag_gini = bag_gini_index(groups, classval)
			if bag_gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], bag_gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

def rf_get_split(dataset,n_features):
 class_values = list(set(row[-1] for row in dataset))
 rf_index, rf_value, rf_score, rf_groups = 999, 999, 999, None
 features = list()
 while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
            for index in features:
                for row in dataset:
                    groups = testsplit(index, row[index], dataset)
                    rf_gini = rf_gini_index(groups, class_values)
 if rf_gini < rf_score:
     rf_index, rf_value, rf_score, rf_groups = index, row[index], rf_gini, groups
 return {'index':rf_index, 'value':rf_value, 'groups':rf_groups}

def toterminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def bag_split(node, maximum_depth, min_size, depth):
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
		node['left'] = bag_get_split(left)
		bag_split(node['left'], maximum_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = toterminal(right)
	else:
		node['right'] = bag_get_split(right)
		bag_split(node['right'], maximum_depth, min_size, depth+1)
        
def rf_split(node, maximum_depth, min_size, n_features, depth):
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
		node['left'] = rf_get_split(left, n_features)
		rf_split(node['left'], maximum_depth, min_size,n_features, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = toterminal(right)
	else:
		node['right'] = rf_get_split(right, n_features)
		rf_split(node['right'], maximum_depth, min_size, n_features, depth+1)
        
def bag_build_tree(train, maximum_depth, min_size):
	root = bag_get_split(train)
	bag_split(root, maximum_depth, min_size, 1)
	return root

def rf_build_tree(train, maximum_depth, min_size, n_features):
	root = rf_get_split(train, n_features)
	rf_split(root, maximum_depth, min_size, n_features, 1)
	return root

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
        
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

def predictionbagging(trees, row):
	rf_predictions = [predict(tree, row) for tree in trees]
	return max(set(rf_predictions), key=rf_predictions.count)

def bagging(bag_train, bag_test, max_depth, min_size, sample_size, n_trees):
	bag_trees = list()
	for i in range(n_trees):
		sample = subsample(bag_train, sample_size)
		bag_tree = bag_build_tree(sample, max_depth, min_size)
		bag_trees.append(bag_tree)
	bag_predictions = [predictionbagging(bag_trees, row) for row in bag_test]
	return(bag_predictions)

def random_forest(rf_train, rf_test, max_depth, min_size, sample_size, n_trees, n_features):
 rf_trees = list()
 for i in range(n_trees):
     sample = subsample(rf_train, sample_size)
     rf_tree = rf_build_tree(sample, max_depth, min_size, n_features)
     rf_trees.append(rf_tree)
 rf_predictions = [predictionbagging(rf_trees, row) for row in rf_test]
 return(rf_predictions)

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
n_features = int(sqrt(len(dataset[0])-1))



for n_trees in [2, 4, 8, 40]:
    bag_scores = eval_algo(dataset, random_forest, n_folds, maximum_depth, min_size, sample_size, n_trees, n_features)
    trees = n_trees
    bag_acc = sum(bag_scores)/float(len(bag_scores))
    print('Trees are: %d' % trees)
    print('Bagging Scores are: %s' % bag_scores)
    print('Mean Accuracy of Bagging is: %.3f%%' % bag_acc)
    rf_scores = eval_algo(dataset, bagging, n_folds, maximum_depth, min_size, sample_size, n_trees)
    trees = n_trees
    rf_acc = sum(rf_scores)/float(len(rf_scores))
    print('Trees are: %d' % trees)
    print('Random Forest Scores are: %s' % rf_scores)
    print('Mean Accuracy of Random Forest is: %.3f%%' % rf_acc)



