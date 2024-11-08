# %%
import pandas as pd 
from scipy.io import mmread
import numpy as np
import matplotlib.pyplot as plt
from dmt import DMT
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import datetime

sns.set_style('dark')

# %%
def read_mtx(x_file_path, y_file_path, dtype='int32', labelencoder=False):
    x = mmread(x_file_path).astype(dtype)
    if labelencoder:
        y = pd.read_csv(y_file_path, sep='\t', header=None)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        y = pd.read_csv(y_file_path, sep='\t', header=None).to_numpy()
    return x, y

def get_data_info(x, y):
    ''''''
    print(f"The size of x: {len(x)}")
    print(f"The shape of x: {x.shape}")
    print(f"The size of y: {len(y)}")
    print(f"The uniques of y: {len(np.unique(y))}")


# get_data_info(X_cel_downsampled, y_cel_em_downsampled) 

dataname = 'covtype'
    
if dataname == 'eminst':
    
    from torchvision.datasets import EMNIST
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])
    train_data = EMNIST(root='data', train=True, download=True, transform=transform, split='byclass')

    DATA = np.stack([train_data[i][0].numpy().squeeze() for i in range(len(train_data))])
    DATA = DATA.reshape((DATA.shape[0],-1))
    y_cel_em = np.array([train_data[i][1] for i in range(len(train_data))])
    
if dataname == 'breast_cancer':
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    DATA = np.array(data.data)  # Feature matrix
    y_cel_em = np.array(data.target)  # Target vector
    
if dataname == 'covtype':
    from sklearn.datasets import fetch_covtype
    cov_type = fetch_covtype()
    # import pdb; pdb.set_trace()
    DATA = np.array(cov_type.data).astype(np.float32)
    y_cel_em = np.array(cov_type.target)  # Target vector

# %%

DATA.shape


# %%

print(f'start dmt, time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
dmt = DMT(num_fea_aim=0.99, device_id=0, epochs=600, batch_size=2000, K=5, nu=1e-2)
# X_cel_dmt, X_cel_umap, X_cel_tsne, X_cel_pca  = dmt.compare(DATA, plot=None)
dmt.fit(DATA)
X_cel_dmt = dmt.transform(DATA)

print(f'start UMAP, time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
import umap
reducer = umap.UMAP()
X_cel_umap = reducer.fit_transform(DATA)

print(f'start TSNE, time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
from sklearn.manifold import TSNE
X_cel_tsne = TSNE(n_components=2).fit_transform(DATA)



# %%
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print(f'start svc, time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
svc_dmt = SVC(max_iter=100000).fit(X_cel_dmt, y_cel_em)
acc_dmt = accuracy_score(y_cel_em, svc_dmt.predict(X_cel_dmt))

svc_umap = SVC(max_iter=100000).fit(X_cel_umap, y_cel_em)
acc_umap = accuracy_score(y_cel_em, svc_umap.predict(X_cel_umap))

svc_tsne = SVC(max_iter=100000).fit(X_cel_tsne, y_cel_em)
acc_tsne = accuracy_score(y_cel_em, svc_tsne.predict(X_cel_tsne))

print('acc_dmt', acc_dmt)
print('acc_umap', acc_umap)
print('acc_tsne', acc_tsne)

# %%

plt.figure(figsize=(20, 7))

# 创建散点图
ax_1 = plt.subplot(1, 3, 3, facecolor='white')
sc1 = ax_1.scatter(X_cel_dmt[:, 0], X_cel_dmt[:, 1], c=y_cel_em, cmap='tab10', s=5)
ax_1.set_title(f'DMT ACC:{acc_dmt}')
ax_1.set_xticks([])  # Remove x-axis labels
ax_1.set_yticks([])  # Remove y-axis labels
ax_1.spines[:].set_color('black')  # Set border color to black
ax_1.spines[:].set_linewidth(1.5)  # Set border thickness

ax_2 = plt.subplot(1, 3, 2, facecolor='white')
sc2 = ax_2.scatter(X_cel_umap[:, 0], X_cel_umap[:, 1], c=y_cel_em, cmap='tab10', s=5)

ax_2.set_title(f'UMAP ACC:{acc_umap}')
ax_2.set_xticks([])  # Remove x-axis labels
ax_2.set_yticks([])  # Remove y-axis labels
ax_2.spines[:].set_color('black')  # Set border color to black
ax_2.spines[:].set_linewidth(1.5)  # Set border thickness

ax_3 = plt.subplot(1, 3, 1, facecolor='white')
sc3 = ax_3.scatter(X_cel_tsne[:, 0], X_cel_tsne[:, 1], c=y_cel_em, cmap='tab10', s=5)

ax_3.set_title(f'TSNE ACC:{acc_tsne}')
ax_3.set_xticks([])  # Remove x-axis labels
ax_3.set_yticks([])  # Remove y-axis labels
ax_3.spines[:].set_color('black')  # Set border color to black
ax_3.spines[:].set_linewidth(1.5)  # Set border thickness

plt.tight_layout()
plt.savefig('covtype.png', dpi=300)

# %%



