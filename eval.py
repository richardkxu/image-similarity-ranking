import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# load embeddings
train_embs = np.load('./train_embs.npy')
train_labels = np.load('./train_labels.npy')
train_paths = np.load('./train_paths.npy')
test_embs = np.load('./test_embs.npy')
test_labels = np.load('./test_labels.npy')
test_paths = np.load('./test_paths.npy')
print("embedding loaded")

# fit knn
neigh = KNeighborsClassifier(n_neighbors=30, n_jobs=12)
neigh.fit(train_embs, train_labels)
print("knn fit finished")

# calculate test dist
time5 = time.time()
dist_test, idx_test = neigh.kneighbors(test_embs, n_neighbors=30, return_distance=True)
time6 = time.time()

np.save('./test_dist', arr=dist_test)
np.save('./test_idx', arr=idx_test)
sec = time6-time5
min, sec = divmod(sec, 60)
hr, min = divmod(min, 60)
print('Top30 test KNN time: {:.2f} hr {:.2f} min {:.2f} sec'.format(hr, min, sec))
print("Test idx shape: {}".format(idx_test.shape))

# calculate test accuracy
test_precision = 0.0
N2 = idx_test.shape[0]
for i in range(N2):
    top30 = train_labels[idx_test[i]]
    true_label = test_labels[i]
    precision = sum(top30 == true_label) / 30.0
    test_precision += precision
test_precision = 100.0 * test_precision / N2
print("Test similarity precision is: {:.3f}%".format(test_precision))

# calculate train dist
time7 = time.time()
dist_train, idx_train = neigh.kneighbors(train_embs, n_neighbors=30, return_distance=True)
time8 = time.time()
np.save('./train_dist', arr=dist_train)
np.save('./train_idx', arr=idx_train)
sec = time8-time7
min, sec = divmod(sec, 60)
hr, min = divmod(min, 60)
print('Top30 train KNN time: {:.2f} hr {:.2f} min {:.2f} sec'.format(hr, min, sec))
print("Train idx shape: {}".format(idx_train.shape))

# calculate training accuracy
train_precision = 0.0
N = idx_train.shape[0]
for i in range(N):
    top30 = train_labels[idx_train[i]]
    true_label = train_labels[i]
    precision = sum(top30 == true_label) / 30.0
    train_precision += precision
train_precision = 100.0 * train_precision / N
print("Training similarity precision is: {:.3f}%".format(train_precision))

