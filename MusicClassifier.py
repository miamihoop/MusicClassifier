import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn import neighbors
import matplotlib.patches as mpatches


genre = pd.read_csv('genre.csv', sep = '\t')
feature = pd.read_csv('features.csv')
df = pd.merge(genre, feature, on=['track_id'])
df = df.dropna(axis=0, how='any')

music_types = df['genre'].unique()
print music_types

mapping = {'Hip-Hop': 1, 'Pop': 9, 'Folk': 2, 'Jazz': 4, 'Rock': 5, 'Electronic': 6, 'International': 7, 'Blues': 8, 'Classical': 3, 'Old-Time / Historic': 10, 'Instrumental': 11, 'Experimental': 12}
df = df.replace({'genre': mapping})

df_hh = df[df['genre'] == 1]
df_fk = df[df['genre'] == 2]
df_cl = df[df['genre'] == 3]
df_sub = df_hh.append(df_fk)
df_sub = df_sub.append(df_cl)

X = df_sub[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']]
# X = df_sub[['danceability', 'energy', 'tempo']]
y = df_sub['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print 'X_train: \n', X_train
print 'X_test: \n', X_test
print 'y_train: \n', y_train
print 'y_test: \n', y_test

cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train, c = y_train)

# train knn model and plot knn boundaries
nn = 15 # nn nearest points
ww = 'distance' # knn type
X_mat = X_train[['danceability', 'energy']].as_matrix()
y_mat = y_train.as_matrix()

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

clf = neighbors.KNeighborsClassifier(n_neighbors = nn, weights = ww)
clf.fit(X_mat, y_mat)
# clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
# clf.fit(X_mat, y_mat)

mesh_step_size = 0.01
plot_symbol_size = 30

x_min, x_max = X_mat[:, 0].min() , X_mat[:, 0].max() 
y_min, y_max = X_mat[:, 1].min() , X_mat[:, 1].max() 
xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                     np.arange(y_min, y_max, mesh_step_size))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot training points
plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y_train, cmap=cmap_bold, edgecolor = 'black')
plt.xlim(0, 1)
plt.ylim(0, 1)

patch0 = mpatches.Patch(color='#FF0000', label='Hip-Hop')
patch1 = mpatches.Patch(color='#00FF00', label='Folk')
patch2 = mpatches.Patch(color='#0000FF', label='Classical')
plt.legend(handles=[patch0, patch1, patch2])

plt.xlabel('danceability')
plt.ylabel('energy')
plt.show()
