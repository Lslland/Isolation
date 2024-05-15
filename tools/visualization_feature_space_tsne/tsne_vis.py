from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

var_output = np.load('res_npy/var_output.npy').astype(np.float64)
var_data = np.load('res_npy/var_data.npy')
var_target = np.load('res_npy/var_target.npy')

poison_output = np.load('res_npy/poison_output.npy').astype(np.float64)
poison_data = np.load('res_npy/poison_data.npy')
poison_target = np.load('res_npy/poison_target.npy')
print(poison_target.shape)

output_2d_var = TSNE(n_components=2, perplexity=30).fit_transform(var_output)
output_2d_poison = TSNE(n_components=2, perplexity=30).fit_transform(poison_output)
# print(output_2d_poison[0:200])
output_2d_var = output_2d_var[4000:6000]
output_2d_poison = output_2d_poison[0:201]
var_target = var_target[4000:6000]

X_var = {i:[] for i in names.keys()}
y_var = {i:[] for i in names.keys()}
num = 200
for i, value in enumerate(var_target):
    if len(y_var[value[0]]) < num:
        y_var[value[0]].append(value)
        X_var[value[0]].append(output_2d_var[i])

print(var_target)

num_class = len(np.unique(var_target))
custom_palette = sns.color_palette("hls", num_class)
print(custom_palette)
cm=mpl.colors.ListedColormap([i for i in custom_palette])
print(np.array(X_var[0]))
print(np.array(y_var[0]))

plt.rcParams['figure.figsize'] = 10, 10
for i, c in enumerate(custom_palette):
    plt.scatter(np.array(X_var[i])[:, 0], np.array(X_var[i])[:, 1], c=c, s=60, alpha=0.6)

plt.scatter(output_2d_poison[:, 0], output_2d_poison[:, 1], c='black', s=60, alpha=0.6)
plt.legend(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck', 'poisoned'])
plt.yticks([])
plt.xticks([])
plt.savefig('visualization/output_2d_isolation_non_iid.png', bbox_inches='tight', dpi=300)
plt.show()

# np.save('res_npy/output_2d_var.npy', output_2d_var, allow_pickle=False)
# plt.rcParams['figure.figsize'] = 10, 10
# plt.scatter(output_2d_var[:, 0], output_2d_var[:, 1], c=var_target, s=60, alpha=0.6, cmap=cm)
# plt.legend(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck'])
#
# np.save('res_npy/output_2d_poison.npy', output_2d_poison, allow_pickle=False)
# plt.rcParams['figure.figsize'] = 10, 10
# plt.scatter(output_2d_poison[:, 0], output_2d_poison[:, 1], c='black', s=60, alpha=0.6)
# # plt.legend(['poisoned'])
#
