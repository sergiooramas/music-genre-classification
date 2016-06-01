import numpy as np
from optparse import OptionParser
import sys
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import random
import json
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn import preprocessing
import nltk
from nltk.tag import map_tag
from collections import Counter
from nltk.stem.porter import PorterStemmer

# parse commandline arguments
op = OptionParser()
op.add_option("--report",
							action="store_true", dest="print_report",
							help="Print a detailed classification report.")
op.add_option("--confusion_matrix",
							action="store_true", dest="print_cm",
							help="Print the confusion matrix.")

(opts, args) = op.parse_args()
if len(args) > 0:
		op.error("this script takes no arguments.")
		sys.exit(1)

print(__doc__)
op.print_help()
print()

load_partition = True
with_entity_features = False
with_text_features = True
with_categories = False
tfidf_features = True
sent_features = True

products = json.load(open("dataset_classification.json","r"))
categories = ['Alternative Rock','Classical','Country','Dance & Electronic','Folk','Jazz','Latin Music','Metal','New Age','Pop','R&B','Rap & Hip-Hop','Rock']
genre_products = dict()
for id, product in products.iteritems():
		genre_products.setdefault(product['genre'],[]).append(id)


def partition(lst, n): 
    division = len(lst) / float(n) 
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]


def create_folds(k,suffix):
	test = []
	train = []
	for i in range(0,k):
		test.append(set())
		train.append(set())
	for genre, ids in genre_products.iteritems():
		rnd = ids[:]
		random.shuffle(rnd)
		folds = partition(rnd, k)
		for i, fold in enumerate(folds):
			test[i].update(fold)
			train[i].update(set(ids).difference(fold))
	for i in range(0,k):
		ftr = open("evaluation/train_"+suffix+str(i)+".csv","w")	
		ftr.write("\n".join(list(train[i])))
		fts = open("evaluation/test_"+suffix+str(i)+".csv","w")	
		fts.write("\n".join(list(test[i])))
	return train, test

def load_folds(k,suffix):
	test = []
	train = []
	for i in range(0,k):
		ftr = open("evaluation/train_"+suffix+str(i)+".csv","r")
		train.append(set(ftr.read().splitlines()))
		fts = open("evaluation/test_"+suffix+str(i)+".csv","r")
		test.append(set(fts.read().splitlines()))
	return train, test

def get_sentiment_count_data(train,test):
	sent_count_train = []
	sent_count_test = []
	v = DictVectorizer(sparse=False)
	for id in test:
		dist = nltk.FreqDist(products[id]['all_pos'].split())
		new_dist = Counter()
		for tag, count in dist.iteritems():
			new_dist[map_tag('en-ptb', 'universal', tag)] += count
		Fscore = 0.5 * ((new_dist['NOUN']+new_dist['ADJ']+new_dist['ADP']+new_dist['DET']) - (dist['UH']+new_dist['VERB']+new_dist['ADV']+new_dist['PRON']) + 100)
		neg_count = 0
		pos_count = 0
		suma = 0
		emotion_words = 0
		for review in products[id]['reviews']:        
			for feature,adjective,score in review['opinions']:
				if score is not None:
					if score < 0:
						neg_count += 1
					else:
						pos_count += 1
					suma += score
					emotion_words += 1
		nwords = len(products[id]['all_text'].split())
		eRatio = emotion_words*1.0/nwords
		posToAllRatio = pos_count*1.0/(pos_count+neg_count)
		emotionFeatures = {'Fscore':Fscore,'eStrength':suma*1.0/emotion_words,'eRatio':eRatio,'posToAllRatio':posToAllRatio}
		sent_count_test.append(emotionFeatures)
	for id in train:
		dist = nltk.FreqDist(products[id]['all_pos'].split())
		new_dist = Counter()
		for tag, count in dist.iteritems():
			new_dist[map_tag('en-ptb', 'universal', tag)] += count
		Fscore = 0.5 * ((new_dist['NOUN']+new_dist['ADJ']+new_dist['ADP']+new_dist['DET']) - (dist['UH']+new_dist['VERB']+new_dist['ADV']+new_dist['PRON']) + 100)
		neg_count = 0
		pos_count = 0
		suma = 0
		emotion_words = 0
		for review in products[id]['reviews']:
			for feature,adjective,score in review['opinions']:
				if score is not None:
					if score < 0:
						neg_count += 1
					else:
						pos_count += 1
					suma += score
					emotion_words += 1
		nwords = len(products[id]['all_text'].split())
		eRatio = emotion_words*1.0/nwords
		posToAllRatio = pos_count*1.0/(pos_count+neg_count)
		emotionFeatures = {'Fscore':Fscore,'eStrength':suma*1.0/emotion_words,'eRatio':eRatio,'posToAllRatio':posToAllRatio}
		sent_count_train.append(emotionFeatures)

	X_sent_train = v.fit_transform(sent_count_train)
	X_sent_test = v.transform(sent_count_test)
	scaler = preprocessing.StandardScaler().fit(X_sent_train)
	X_train = scaler.transform(X_sent_train)
	X_test = scaler.transform(X_sent_test)

	return sent_count_train, sent_count_test, X_train, X_test

def get_semantic_data(train,test,broaders=False):
	entities = json.load(open("semantic_features.json"))
	data_train = []
	data_test = []
	for id in train:
		if broaders:
			data_train.append(" ".join([str(e) for e in entities[id]['entities']])+" ".join(entities[id]['categories'])+" ".join(entities[id]['broaders']))
		else:
			data_train.append(" ".join([str(e) for e in entities[id]['entities']])+" ".join(entities[id]['categories']))
	for id in test:
		if broaders:
			data_test.append(" ".join([str(e) for e in entities[id]['entities']])+" ".join(entities[id]['categories'])+" ".join(entities[id]['broaders']))
		else:
			data_test.append(" ".join([str(e) for e in entities[id]['entities']])+" "+" ".join(entities[id]['categories']))
	sem_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
	X_sem_train = sem_vectorizer.fit_transform(data_train)
	X_sem_test = sem_vectorizer.transform(data_test)
	return data_train, data_test, X_sem_train, X_sem_test

def get_acoustic_data(train,test):
	folder = "flattened_acoustic_descriptors/"
	v = DictVectorizer(sparse=False)
	data_train = []
	data_test = []
	for id in train:
		features = json.load(open(folder+id+".json"))
		data_train.append(features)
	for id in test:
		features = json.load(open(folder+id+".json"))
		data_test.append(features)
	X_train = v.fit_transform(data_train)
	X_test = v.transform(data_test)
	scaler = preprocessing.StandardScaler(with_mean=False).fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	return data_train, data_test, X_train, X_test

stemmer = PorterStemmer()

def classify(train,test,features):	
	y_train = [products[id]['genre'] for id in train]
	y_test = [products[id]['genre'] for id in test]

	X_train_d = dict()
	X_test_d = dict()

	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', ngram_range=(1,2), analyzer='word')
	if 'bow' in features:
		data_train = [products[id]['all_text'] for i,id in enumerate(train)]
		data_test = [products[id]['all_text'] for i,id in enumerate(test)]
		X_train_d['bow'] = vectorizer.fit_transform(data_train)
		X_test_d['bow'] = vectorizer.transform(data_test)
	if 'sent-count' in features:
		data_sent_train, data_sent_test, X_train_d['sent-count'], X_test_d['sent-count'] = get_sentiment_count_data(train,test)
	if 'semantic' in features:
		data_sem_train, data_sem_test, X_train_d['semantic'], X_test_d['semantic'] = get_semantic_data(train,test,False)
	if 'semantic-broader' in features:
		data_sem_train, data_sem_test, X_train_d['semantic-broader'], X_test_d['semantic-broader'] = get_semantic_data(train,test,True)
	if 'acoustic' in features:
		data_ac_train, data_ac_test, X_train_d['acoustic'], X_test_d['acoustic'] = get_acoustic_data(train,test)

	X_train = X_train_d[features[0]]
	X_test = X_test_d[features[0]]
	for i in range(1,len(features)):		
		X_train = hstack((X_train,X_train_d[features[i]]),format='csr')
		X_test = hstack((X_test,X_test_d[features[i]]),format='csr')

	def trim(s):
			"""Trim string to fit on terminal (assuming 80-column display)"""
			return s if len(s) <= 80 else s[:77] + "..."

	###############################################################################
	# Benchmark classifiers
	def benchmark(clf):
			print('_' * 80)
			print("Training: ")
			#print(clf)
			t0 = time()
			clf.fit(X_train, y_train)
			train_time = time() - t0
			print("train time: %0.3fs" % train_time)

			t0 = time()
			pred = clf.predict(X_test)
			test_time = time() - t0
			print("test time:  %0.3fs" % test_time)

			score = metrics.accuracy_score(y_test, pred)
			print("accuracy:   %0.3f" % score)

			if hasattr(clf, 'coef_'):
					print("dimensionality: %d" % clf.coef_.shape[1])
					print("density: %f" % density(clf.coef_))

			if opts.print_report:
					print("classification report:")
					print(metrics.classification_report(y_test, pred,
																							target_names=categories))

			if opts.print_cm:
					print("confusion matrix:")
					print(metrics.confusion_matrix(y_test, pred))

			print()
			clf_descr = str(clf).split('(')[0]
			return clf_descr, score, train_time, test_time, metrics.confusion_matrix(y_test, pred)


	results = []
	for clf, name in (
					(RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
					(RandomForestClassifier(n_estimators=100), "Random forest")):
			print('=' * 80)
			print(name)
			results.append(benchmark(clf))

	for penalty in ["l2"]:
			print('=' * 80)
			print("%s penalty" % penalty.upper())
			# Train Liblinear model
			results.append(benchmark(LinearSVC(loss='squared_hinge', penalty=penalty, dual=False, tol=1e-3)))
	# Train NearestCentroid without threshold
	print('=' * 80)
	print("NearestCentroid (aka Rocchio classifier)")
	results.append(benchmark(NearestCentroid()))

	print([r[0]+" "+str(r[1]) for r in results])
	# make some plots

	return [(r[0],r[1],r[4]) for r in results]


if __name__ == '__main__':
	train,test = load_folds(5,"")
	#train,test = create_folds(5,"reviews")
	experiments = [['bow'],['bow','semantic'],['bow','semantic-broader'],['bow','sent-count'],['bow','semantic','sent-count']]
	for features in experiments:
		results = dict()
		fw = open("evaluation/results","a")
		fw.write("+".join(features)+"\n")
		confusion_matrix = np.zeros((13,13))
		for i, (train_i, test_i) in enumerate(zip(train,test)):
			res = classify(train_i, test_i, features)
			for classifier, precision, matrix in res:
				results.setdefault(classifier, []).append(precision)
				confusion_matrix += np.matrix(matrix)
		for classifier, res in results.iteritems():
			print classifier, np.mean(res), np.std(res)
			fw.write(classifier+" "+str(np.mean(res))+" "+str(np.std(res))+" "+str(max(res, key=float) )+" "+str(min(res, key=float) )+"\n")
		fw.write("\n")
		print confusion_matrix
			
