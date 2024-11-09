# %% Import required libraries
import pandas as pd
from scipy.io import mmread
import numpy as np
import matplotlib.pyplot as plt
from dmt import DMT
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import datetime

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
    print("Reading data from mtx file...")
    x = mmread(x_file_path).astype(dtype)
    if labelencoder:
        y = pd.read_csv(y_file_path, sep='\t', header=None)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print("Label encoding applied to y.")
    else:
        y = pd.read_csv(y_file_path, sep='\t', header=None).to_numpy()
    print("Data loading complete.")
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
dataname = 'minst'

if dataname == 'minst':
    print("Loading MNIST dataset...")
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    # Set up data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])
    # Load the MNIST dataset
    train_data = MNIST(root='data', train=True, download=True, transform=transform)
    DATA = np.stack([train_data[i][0].numpy().squeeze() for i in range(len(train_data))]).reshape((-1, 784))
    y_cel_em = np.array([train_data[i][1] for i in range(len(train_data))])
    print("MNIST dataset loaded successfully.")

elif dataname == 'eminst':
    print("Loading EMNIST dataset...")
    from torchvision.datasets import EMNIST
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = EMNIST(root='data', train=True, download=True, transform=transform, split='byclass')
    DATA = np.stack([train_data[i][0].numpy().squeeze() for i in range(len(train_data))]).reshape((-1, 784))
    y_cel_em = np.array([train_data[i][1] for i in range(len(train_data))])
    print("EMNIST dataset loaded successfully.")

elif dataname == 'breast_cancer':
    print("Loading Breast Cancer dataset...")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    DATA, y_cel_em = np.array(data.data), np.array(data.target)
    print("Breast Cancer dataset loaded successfully.")

# %% Display data shape
print("Dataset information:")
print("Data shape:", DATA.shape)
get_data_info(DATA, y_cel_em)

# %% Dimensionality reduction using DMT, UMAP, and TSNE
print(f"Starting DMT, time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
dmt = DMT(num_fea_aim=0.99, device_id=0, epochs=300, batch_size=2000, K=5, nu=1e-2)
X_cel_dmt, X_cel_umap, X_cel_tsne, X_cel_pca = dmt.compare(DATA, plot=None)
print(f"DMT completed, time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"Starting UMAP, time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
reducer = umap.UMAP()
X_cel_umap = reducer.fit_transform(DATA)
print(f"UMAP completed, time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"Starting TSNE, time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
X_cel_tsne = TSNE(n_components=2).fit_transform(DATA)
print(f"TSNE completed, time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% SVM classification and accuracy calculation for each embedding method
print("Starting SVM classification...")
svc_dmt = SVC().fit(X_cel_dmt, y_cel_em)
acc_dmt = accuracy_score(y_cel_em, svc_dmt.predict(X_cel_dmt))
print(f'Accuracy with DMT: {acc_dmt}')

svc_umap = SVC().fit(X_cel_umap, y_cel_em)
acc_umap = accuracy_score(y_cel_em, svc_umap.predict(X_cel_umap))
print(f'Accuracy with UMAP: {acc_umap}')

svc_tsne = SVC().fit(X_cel_tsne, y_cel_em)
acc_tsne = accuracy_score(y_cel_em, svc_tsne.predict(X_cel_tsne))
print(f'Accuracy with TSNE: {acc_tsne}')

# %% Visualization of embeddings with accuracy in titles
print("Creating visualizations...")
plt.figure(figsize=(20, 7))

# Plot DMT results
ax_1 = plt.subplot(1, 3, 3, facecolor='white')
ax_1.scatter(X_cel_dmt[:, 0], X_cel_dmt[:, 1], c=y_cel_em, cmap='tab10', s=0.5)
ax_1.set_title(f'DMT ACC: {acc_dmt}')
ax_1.set_xticks([]); ax_1.set_yticks([])
ax_1.spines[:].set_color('black'); ax_1.spines[:].set_linewidth(1.5)

# Plot UMAP results
ax_2 = plt.subplot(1, 3, 2, facecolor='white')
ax_2.scatter(X_cel_umap[:, 0], X_cel_umap[:, 1], c=y_cel_em, cmap='tab10', s=0.5)
ax_2.set_title(f'UMAP ACC: {acc_umap}')
ax_2.set_xticks([]); ax_2.set_yticks([])
ax_2.spines[:].set_color('black'); ax_2.spines[:].set_linewidth(1.5)

# Plot TSNE results
ax_3 = plt.subplot(1, 3, 1, facecolor='white')
ax_3.scatter(X_cel_tsne[:, 0], X_cel_tsne[:, 1], c=y_cel_em, cmap='tab10', s=0.5)
ax_3.set_title(f'TSNE ACC: {acc_tsne}')
ax_3.set_xticks([]); ax_3.set_yticks([])
ax_3.spines[:].set_color('black'); ax_3.spines[:].set_linewidth(1.5)

# Final layout adjustments and save the figure
plt.tight_layout()
plt.savefig('mnist.png', dpi=300)
print("Visualizations saved as 'mnist.png'.")