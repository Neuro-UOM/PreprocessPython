# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

path ='./Data/SDK/'
#all_files = os.path.join(path , "*.csv")
#print all_files
#
#df = pd.concat((pd.read_csv(f, encoding='utf-16', header=None) for f in all_files))

allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)

#check nulls
#print pd.isnull(frame).any()

#s = pd.Series(frame['Label'], dtype="category")
#print s

#frame colomns list
#print frame.columns.values.tolist()

#rows of 
#print frame['16 Gamma'].shape

###############################################################
# nothing useful
#data_correlations = frame.corr()
#
#
## plot co relation
#sj_corr_heat = sns.heatmap(data_correlations)
#plt.title('correlations')
###############################################################

features = []
waves = ["Low_beta","High_beta","Alpha","Theta", "Gamma"]
for i in range(7,13):
    for j in waves:
        features.append(str(i)+ " "+ j)

features.append("Label")
print features

frame = frame[features]
frame['Label'] = frame['Label'].map({'null': 1, 'green': 0, 'red': 0})
frame = frame[frame.Label != 2]

#s = pd.Series(frame['Label'], dtype="category")
#print s

#print frame['Label']

#data_correlations = frame.corr()

# plot co relation
#corr_heat = sns.heatmap(data_correlations)
#plt.title('correlations')

#(data_correlations
#     .Label
#     .drop('Label') # don't compare with myself
#     .sort_values(ascending=False)
#     .plot
#     .barh())

frame.to_csv('col_null.csv')

##################################################################
# pca clustering

#from sklearn.cluster import DBSCAN
#from sklearn import metrics
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.preprocessing import StandardScaler
#
#labels_true = frame['Label']
#
#ft = ['10 Alpha', '11 Alpha']
#c_frame = frame[ft]
#
#X = StandardScaler().fit_transform(c_frame)
#
#db = DBSCAN(eps=0.3, min_samples=10).fit(X)
#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True
#labels = db.labels_
#
## Number of clusters in labels, ignoring noise if present.
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#
#print('Estimated number of clusters: %d' % n_clusters_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(labels_true, labels))
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X, labels))
#
#
## Black removed and is used for noise instead.
#unique_labels = set(labels)
#colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
#for k, col in zip(unique_labels, colors):
#    if k == -1:
#        # Black used for noise.
#        col = 'k'
#
#    class_member_mask = (labels == k)
#
#    xy = X[class_member_mask & core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=14)
#
#    xy = X[class_member_mask & ~core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=6)
#
#plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.show()

#############################################################################
# neural network classification

#from sklearn.neural_network import MLPClassifier
#
#print frame.shape
#
#data_subtrain = frame.head(2000)
#data_subtest = frame.tail(frame.shape[0] - 2000)
#
##data_subtest = data_subtest.drop('Label', axis=1, inplace=True)
#
#ft = ['10 Alpha', '11 Alpha','9 Alpha','10 High_beta', '10 Low_beta','Label']
#c_frame = data_subtrain[ft]
#
#ft2 = ['10 Alpha', '11 Alpha','9 Alpha','10 High_beta', '10 Low_beta','Label']
#test_frame = data_subtest[ft2]
#
#X = c_frame
#Y = c_frame['Label']
#
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                    hidden_layer_sizes=(5, 2), random_state=1)
#
#clf.fit(X, Y)
#
#pred = clf.predict(test_frame)
#
#print pred

