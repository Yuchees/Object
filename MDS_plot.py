#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import manifold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AffinityPropagation


def data_retrieve(path, normalize):
	global df_structure, normalized_df
	file_format = path.split('.')[-1]
	if file_format == 'csv':
		df = pd.read_csv(path)
	elif file_format == ('xls' or 'xlsx'):
		df = pd.read_excel(path)
	else:
		print('Please input the correct format file.')
	for column in df.columns:
		if column == 'Structure':
			df_structure = pd.DataFrame(df[column])
		if df.dtypes[column] != ('float64' or 'int64'):
			df = df.drop(column, axis=1)
	if normalize == 'mean':
		normalized_df = (df - df.mean()) / (df.max() - df.min())
	elif normalize == 'min_max':
		normalized_df = (df - df.min()) / (df.max() - df.min())
	else:
		exit()
	features = normalized_df.iloc[:, :-1].values
	return features, df_structure


def distance_calculation(data):
	distance = euclidean_distances(data)
	similarities = distance / distance.max()
	return similarities


def merge_data_frames(df1, df2):
	df = pd.merge(df1, df2, left_index=True, right_index=True)
	return df


def mds_cal(similarities):
	print('Start...')
	start = datetime.now()
	seed = np.random.RandomState(seed=0)
	mds = manifold.MDS(
		n_components=2, max_iter=30000, random_state=seed,
		eps=1e-12, dissimilarity="precomputed", n_jobs=1
	)
	pos = mds.fit(similarities).embedding_
	print('Finished. Total time:{}'.format(datetime.now() - start))
	return pos


def affinity_propagation_cluster(features):
	print('Start...')
	start = datetime.now()
	af = AffinityPropagation(
		max_iter=5000, convergence_iter=50, preference=None
	).fit(features)
	cluster_centers_indices = af.cluster_centers_indices_
	labels = pd.DataFrame(af.labels_).rename(
		columns={0: 'AffinityPropagation'}, inplace=True
	)
	n_clusters_ = len(cluster_centers_indices)
	print('Estimated number of clusters: %d' % n_clusters_)
	print('Finished. Total time:{}'.format(datetime.now() - start))
	return labels


if __name__ == '__main__':
	data_retrieve(path='data.csv', normalize='mean')
