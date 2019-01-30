import numpy as np


test_idx = np.load('./test_idx_50.npy')
test_dist = np.load('./test_dist_50.npy')
train_labels = np.load('./train_labels.npy')
test_labels = np.load('./test_labels.npy')
train_paths = np.load('./train_paths.npy')
test_paths = np.load('./test_paths.npy')

top10_idx = test_idx[:, :10]
top10_dist = test_dist[:, :10]
bottom10_idx = test_idx[:, -10:]
bottom10_dist = test_dist[:, -10:]


N = top10_idx.shape[0]
for i in range(N):
    print("test img: {}".format(test_paths[i]))
    print("test label: {}".format(test_labels[i]))
    top10 = top10_idx[i]
    print("top10 labels:")
    print(train_labels[top10])
    print("top10 dist:")
    print(top10_dist[i])
    print("top10 paths:")
    print(train_paths[top10])

    bottom10 = bottom10_idx[i]
    print("bottom10 labels:")
    print(train_labels[bottom10])
    print("bottom10 dist:")
    print(bottom10_dist[i])
    print("bottom10 paths:")
    print(train_paths[bottom10])
    print("\n")

