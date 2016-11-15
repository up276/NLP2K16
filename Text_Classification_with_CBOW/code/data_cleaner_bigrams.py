#! /usr/bin/env python

##########################################################################################################
# PURPOSE OF THE FILE
# INPUT : Input files 
# 1) find most frequent words and generate new files in which non frequent words are replaced by '<oov>' (out of vocab)
# 2) create ngrams
#################################################################################################################
import numpy as np
import os
from data_helpers import clean_str, load_data
from collections import Counter

# only take forst 400 to avoid 0s in embedding
cut_review = 400

#---------------------- CALCULATE FREQUENT WORDS------------------------------------#
train_pos_path = './data/aclImdb/train/pos/'
train_neg_path = './data/aclImdb/train/neg/'
test_pos_path = './data/aclImdb/test/pos/'
test_neg_path = './data/aclImdb/test/neg/'

train_positive_examples = load_data(train_pos_path)
train_negative_examples = load_data(train_neg_path)

# Clean review
train_reviews = train_positive_examples + train_negative_examples
train_reviews = [clean_str(sentence, cut_review) for sentence in train_reviews]

# Build the list of frequent words
word_list = [word for line in train_reviews for word in line.split()]
frequent_words_count = Counter(word_list).most_common(10000) # generates list of tuples (word, word_count) and only keeps the 10000 common ones
frequent_words = [item[0] for item in frequent_words_count] 
very_frequent_words_count = Counter(word_list).most_common(5000)
very_frequent_words = [item[0] for item in very_frequent_words_count] 


#--------------------------------- update  files with '<oov>' for other infrequent words------------#



def replacebyoov(review):
	sequence = review.split()
	for word in sequence:
		if word.lower() in frequent_words:
			continue 
		else:
			sequence[sequence.index(word)] = "<oov>" 
	return ' '.join(sequence)



def clean_file(input_path, output_path, cut_review):
	'''
	from an input file, output the same file but replace unfrequent words from corpus by '<oov>' (out of vocabulary)
	'''
	for filename in os.listdir(input_path):
		
		review = open(input_path+filename, "r").readlines()[0].strip()
         	review = clean_str(review, cut_review)  # clean + cut review after cut_review words	
		review = replacebyoov(review)
		bigrams =  createBiGrams(review)  # Create bigrams
		review_with_bigrams = ' '.join([review, bigrams])
		clean_review = open(output_path + 'clean_' + filename, "w")
		clean_review.write(review_with_bigrams)		
		clean_review.close()
         

clean_train_pos_path = './data/bigrams_400/train/pos/'
clean_train_neg_path = './data/bigrams_400/train/neg/'
clean_test_pos_path = './data/bigrams_400/test/pos/'
clean_test_neg_path = './data/bigrams_400/test/neg/'   
# Create the new files using above paths
clean_file(train_pos_path, clean_train_pos_path, cut_review)
clean_file(train_neg_path, clean_train_neg_path, cut_review)
clean_file(test_pos_path, clean_test_pos_path, cut_review)
clean_file(test_neg_path, clean_test_neg_path, cut_review)








