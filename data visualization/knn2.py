import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import shutil
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_images(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            images.append(np.array(img).flatten())  # Flatten the image to 1D
            filenames.append(filename)
    return np.array(images), filenames

directory = '../data/classes/2'
images, filenames = load_images(directory)

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(images)

# Assign images to clusters
clustered_images = {i: [] for i in range(n_clusters)}
for idx, cluster_id in enumerate(clusters):
    clustered_images[cluster_id].append(filenames[idx])

# Create directories for each cluster
for cluster_id in range(n_clusters):
    cluster_dir = os.path.join(directory, f'cluster_{cluster_id}')
    os.makedirs(cluster_dir, exist_ok=True)

# Move images to respective cluster directories
for cluster_id, image_files in clustered_images.items():
    cluster_dir = os.path.join(directory, f'cluster_{cluster_id}')
    for image_file in image_files:
        src_path = os.path.join(directory, image_file)
        dst_path = os.path.join(cluster_dir, image_file)
        shutil.move(src_path, dst_path)

# PCA reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(images)

# t-SNE reduction
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_result = tsne.fit_transform(images)

# Clustering
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(images)

import matplotlib.pyplot as plt

def plot_clusters(result, clusters, title='PCA result'):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(result[:, 0], result[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

# Plot PCA and t-SNE results
plot_clusters(pca_result, clusters, title='PCA Result')
plot_clusters(tsne_result, clusters, title='t-SNE Result')

# import matplotlib.pyplot as plt

# def show_images_from_cluster(cluster_id, n=5):
#     sample_files = np.random.choice(clustered_images[cluster_id], size=n, replace=False)
#     fig, axs = plt.subplots(1, n, figsize=(15, 3))
#     for i, file in enumerate(sample_files):
#         img_path = os.path.join(directory, file)
#         img = Image.open(img_path)
#         axs[i].imshow(img, cmap='gray')
#         axs[i].axis('off')
#         axs[i].set_title(f'Cluster {cluster_id}')
#     plt.show()

# # Display images from each cluster
# for i in range(n_clusters):
#     show_images_from_cluster(i)
