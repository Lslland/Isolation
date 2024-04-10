from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

output = np.load('res_npy/output.npy').astype(np.float64)
data = np.load('res_npy/data.npy')
target = np.load('res_npy/target.npy')
print('data shape: ', data.shape)
print('target shape: ', target.shape)
print('output shape: ', output.shape)

output_2d = TSNE(n_components=2, perplexity=30).fit_transform(output)
np.save('res_npy/output_2d.npy', output_2d, allow_pickle=False)
plt.rcParams['figure.figsize'] = 20, 20
plt.scatter(output_2d[:, 0], output_2d[:, 1], c=target)
plt.savefig('visualization/output_2d.png', bbox_inches='tight')
plt.show()