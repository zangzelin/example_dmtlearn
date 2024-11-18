# %% Import required libraries
import pandas as pd
from scipy.io import mmread
import numpy as np
import matplotlib.pyplot as plt
from dmt import DMT
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set seaborn style for plots
sns.set_style('dark')

# %% Define functions for data processing
def read_mtx(x_file_path, y_file_path, dtype='int32', labelencoder=False):
    """
    Reads data from mtx file and optional label file.
    
    Parameters:
        x_file_path (str): Path to features file in mtx format.
        y_file_path (str): Path to labels file in text format.
        dtype (str): Data type for features.
        labelencoder (bool): If True, applies LabelEncoder to labels.
    
    Returns:
        tuple: Feature matrix (x) and labels (y).
    """
    x = mmread(x_file_path).astype(dtype)
    if labelencoder:
        y = pd.read_csv(y_file_path, sep='\t', header=None)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        y = pd.read_csv(y_file_path, sep='\t', header=None).to_numpy()
    return x, y

def get_data_info(x, y):
    """
    Displays information about the dataset.
    
    Parameters:
        x (array): Feature matrix.
        y (array): Labels.
    """
    print(f"The size of x: {len(x)}")
    print(f"The shape of x: {x.shape}")
    print(f"The size of y: {len(y)}")
    print(f"The unique values in y: {len(np.unique(y))}")

# %% Load dataset based on selected dataset name
dataname = 'iris'

if dataname == 'iris':
    from sklearn import datasets
    iris = datasets.load_iris()
    DATA = np.array(iris.data)
    y_cel_em = np.array(iris.target)

# elif dataname == 'breast_cancer':
#     from sklearn.datasets import load_breast_cancer
#     data = load_breast_cancer()
#     DATA, y_cel_em = np.array(data.data), np.array(data.target)

# %% Display data shape
print("Data shape:", DATA.shape)

# %% Dimensionality reduction using DMT and other methods
dmt = DMT(num_fea_aim=0.99, device_id=0, epochs=600, batch_size=2000, K=5, nu=1e-2)
X_cel_dmt, X_cel_umap, X_cel_tsne, X_cel_pca = dmt.compare(DATA, plot=None)

# %% SVM classification and accuracy calculation for each embedding method
print(f"Starting SVC, time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
svc_dmt = SVC(max_iter=100000).fit(X_cel_dmt, y_cel_em)
acc_dmt = accuracy_score(y_cel_em, svc_dmt.predict(X_cel_dmt))

svc_umap = SVC(max_iter=100000).fit(X_cel_umap, y_cel_em)
acc_umap = accuracy_score(y_cel_em, svc_umap.predict(X_cel_umap))

svc_tsne = SVC(max_iter=100000).fit(X_cel_tsne, y_cel_em)
acc_tsne = accuracy_score(y_cel_em, svc_tsne.predict(X_cel_tsne))

print('Accuracy with DMT:', acc_dmt)
print('Accuracy with UMAP:', acc_umap)
print('Accuracy with TSNE:', acc_tsne)

# %% Visualization of embeddings with accuracy in titles
plt.figure(figsize=(20, 7))

# Plot DMT results
ax_1 = plt.subplot(1, 3, 3, facecolor='white')
ax_1.scatter(X_cel_dmt[:, 0], X_cel_dmt[:, 1], c=y_cel_em, cmap='tab10', s=15)
ax_1.set_title(f'DMT ACC: {acc_dmt}')
ax_1.set_xticks([]); ax_1.set_yticks([])
ax_1.spines[:].set_color('black'); ax_1.spines[:].set_linewidth(1.5)

# Plot UMAP results
ax_2 = plt.subplot(1, 3, 2, facecolor='white')
ax_2.scatter(X_cel_umap[:, 0], X_cel_umap[:, 1], c=y_cel_em, cmap='tab10', s=15)
ax_2.set_title(f'UMAP ACC: {acc_umap}')
ax_2.set_xticks([]); ax_2.set_yticks([])
ax_2.spines[:].set_color('black'); ax_2.spines[:].set_linewidth(1.5)

# Plot TSNE results
ax_3 = plt.subplot(1, 3, 1, facecolor='white')
ax_3.scatter(X_cel_tsne[:, 0], X_cel_tsne[:, 1], c=y_cel_em, cmap='tab10', s=15)
ax_3.set_title(f'TSNE ACC: {acc_tsne}')
ax_3.set_xticks([]); ax_3.set_yticks([])
ax_3.spines[:].set_color('black'); ax_3.spines[:].set_linewidth(1.5)

# Final layout adjustments and save the figure
plt.tight_layout()
plt.savefig('iris.png', dpi=300)