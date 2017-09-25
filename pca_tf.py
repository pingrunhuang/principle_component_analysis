import tensorflow as tf
import numpy as np
from numba import jit

class TF_PCA():
	
	def __init__(self, data, target=None, dtype=tf.float32):
		self.data = data
		self.target = target
		self.dtype = dtype

		self.graph = None
		self.X = None
		self.U = None
		self.sigma = None
		self.singular_values = None

	# fit is to create U and sigma for later use
	def fit(self):
		# create a graph
		self.graph = tf.Graph()
		# just create a skeleton
		with self.graph.as_default():
			self.X = tf.placeholder(self.dtype, shape=self.data.shape)
			singular_values, u, v = tf.svd(self.X)
			sigma = tf.diag(singular_values)

		with tf.Session(graph=self.graph) as session:
			self.U, self.sigma, self.singular_values = session.run([u, sigma, singular_values], feed_dict={self.X:self.data})

	
	def reduce(self, dimensions=None, keep_info=None):
		if dimensions == None and keep_info == None:
			raise ValueError("Either dimensions or keep_info should be supplied to use this funcion")

		# this procedure is basically getting the number of dimension to be kept
		if keep_info:
			# normalize the singular values
			normalized_singular_values = self.singular_values/sum(self.singular_values)
			# perform cumulative sum which return an array 
			cumulative_sum = np.cumsum(normalized_singular_values)
			# get the index where the cumulative sum is just larger then the required info value. adding 1 because it is zero based
			index = next(i for i, v in enumerate(cumulative_sum) if v >= keep_info) + 1
			dimensions = index

		# define the computation
		with self.graph.as_default():
			# slice the self sigma matrix from [0,0] to 
			sigma = tf.slice(self.sigma, [0,0], [self.sigma.shape[1], dimensions])
			pca = tf.matmul(self.U, sigma)

		with tf.Session(graph=self.graph) as session:
			return session.run(pca, feed_dict={self.X: self.data})


from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset = pd.read_csv('data/breast-cancer-wisconsin.data')
dataset = dataset.replace('?', np.nan).dropna()
data = dataset.iloc[:,1:10]
target = dataset.iloc[:, 10].astype(int)

tf_pca = TF_PCA(data, target)

# iris_data = datasets.load_iris()
# print(iris_data.data)
# print(iris_data.target)
# tf_pca = TF_PCA(iris_data.data, iris_data.target)


tf_pca.fit()
pca = tf_pca.reduce(keep_info=0.9)  # Results in 2 dimensions

color_mapping = {0: sns.xkcd_rgb['bright purple'], 1: sns.xkcd_rgb['lime'], 2: sns.xkcd_rgb['ochre']}
color_mapping = {2: sns.xkcd_rgb['bright purple'], 4: sns.xkcd_rgb['lime']}
colors = list(map(lambda x: color_mapping[x], tf_pca.target))

plt.scatter(pca[:, 0], pca[:, 1], c=colors)
plt.show()

