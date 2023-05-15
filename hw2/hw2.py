import cv2
import numpy as np
import math
import os

def read_image_and_cells(image_file, txt_file):
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    cells = []
    with open(txt_file, 'r') as file:
        for line in file.readlines():
            x, y, cell_type = line.split()
            cells.append((int(x), int(y), cell_type))
    return img, cells

def calculateIntensityFeatures(patch, binNumber):
    mean = np.mean(patch)
    std_dev = np.std(patch)
    
    hist, _ = np.histogram(patch, bins=binNumber)
    probabilities = hist / np.sum(hist)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    intensityFeat = (mean, std_dev, entropy)
    return intensityFeat

def calculateCooccurrenceMatrix(patch, binNumber, di, dj):
    # Discretize the patch into bins
    discretized_patch = np.floor((patch / 256) * binNumber).astype(int)
    
    # Initialize an empty co-occurrence matrix
    M = np.zeros((binNumber, binNumber), dtype=int)

    # Iterate through the patch and update the co-occurrence matrix
    rows, cols = patch.shape
    row_start = max(0, -di)
    row_end = rows + min(0, -di)
    col_start = max(0, -dj)
    col_end = cols + min(0, -dj)
    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            intensity_i = discretized_patch[i, j]
            intensity_j = discretized_patch[i + di, j + dj]
            M[intensity_i, intensity_j] += 1          
    return M


def calculateAccumulatedCooccurrenceMatrix(patch, binNumber, d):
    # Define the distance pairs for which co-occurrence matrices will be calculated
    distance_pairs = [(d, 0), (d, d), (0, d), (-d, d), (-d, 0), (-d, -d), (0, -d), (d, -d)]

    # Initialize an empty accumulated co-occurrence matrix
    accM = np.zeros((binNumber, binNumber), dtype=int)

    # Calculate the co-occurrence matrix for each distance pair and accumulate the results
    for di, dj in distance_pairs:
        M = calculateCooccurrenceMatrix(patch, binNumber, di, dj)
        accM += M

    return accM

def calculateCooccurrenceFeatures(accM):
    # Normalize the accumulated co-occurrence matrix
    normM = accM / accM.sum()
    
    # Initialize the features
    angular_second_moment = 0
    maximum_probability = 0
    inverse_difference_moment = 0
    entropy = 0
    
    # Calculate the features
    for i in range(normM.shape[0]):
        for j in range(normM.shape[1]):
            angular_second_moment += normM[i, j] ** 2
            maximum_probability = max(maximum_probability, normM[i, j])
            inverse_difference_moment += normM[i, j] / (1 + (i - j) ** 2)
            if normM[i, j] > 0:
                entropy += -normM[i, j] * np.log2(normM[i, j])
    
    texturalFeat = (angular_second_moment, maximum_probability, inverse_difference_moment, entropy)
    return texturalFeat

def normalize_features(features):
    feature_array = np.array(features)
    means = feature_array.mean(axis=0)
    stds = feature_array.std(axis=0)
    stds[stds == 0] = 1e-10  # Replace zero standard deviations with a small positive number
    normalized_features = (feature_array - means) / stds
    return normalized_features

def k_means_clustering(features, k, n_init=10, max_iters=100, tol=1e-4):
    best_inertia = None
    best_centroids = None
    best_labels = None

    for _ in range(n_init):
        centroids = features[np.random.choice(features.shape[0], k, replace=False)]
        for _ in range(max_iters):
            distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([features[labels == i].mean(axis=0) for i in range(k)])

            if np.linalg.norm(new_centroids - centroids) < tol:
                break

            centroids = new_centroids

        inertia = np.sum([np.sum(distances[labels == i] ** 2) for i in range(k)])
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels

# Define the function to calculate ratios
def calculate_ratios(cells, cluster_labels, k):
    # Initialize the counts for each cell type in each cluster
    counts = np.zeros((k, 3))

    # Count the cell types in each cluster
    for cell, label in zip(cells, cluster_labels):
        cell_type = cell[2]
        if cell_type == 'inflammatory':
            counts[label, 0] += 1
        elif cell_type == 'epithelial':
            counts[label, 1] += 1
        else:  # cell_type == 'spindle-shaped'
            counts[label, 2] += 1

    # Calculate the ratios
    ratios = counts / counts.sum(axis=1, keepdims=True)
    return ratios

# Define the function to visualize clusters
def visualize_clusters(image_file, cells, cluster_labels, k, colors):
    img = cv2.imread(image_file)
    for cell, label in zip(cells, cluster_labels):
        x, y, _ = cell
        cv2.circle(img, (x, y), 5, colors[label], -1)
    return img
images_path = "/Users/derinberktay/Desktop/448/hw2/nucleus-dataset/images/"
image_files = sorted([file for file in os.listdir(images_path) if file.endswith(".png")])

def mainfunction(binNumber, d, N, k):  
    txt_files_path = "/Users/derinberktay/Desktop/448/hw2/nucleus-dataset/txt_files/"  
    txt_files = sorted([file for file in os.listdir(txt_files_path) if file.endswith("_cells")])
    all_features = []

    for image_file, txt_file in zip(image_files, txt_files):
        image_path = os.path.join(images_path, image_file)
        txt_path = os.path.join(txt_files_path, txt_file)
        img, cells = read_image_and_cells(image_path, txt_path)
        for cell in cells:
            x, y, cell_type = cell
            patch = img[y-N//2:y+N//2, x-N//2:x+N//2]  # Use N instead of hardcoded value
            intensity_features = calculateIntensityFeatures(patch, binNumber)
            #cooccurrence_matrix = calculateCooccurrenceMatrix(patch, 8, 1, 1)
            accumulated_cooccurance_matrix = calculateAccumulatedCooccurrenceMatrix(patch, binNumber, d)
            coocurance_features = calculateCooccurrenceFeatures(accumulated_cooccurance_matrix)
            all_features.append(np.concatenate((intensity_features, coocurance_features)))

    normalized_features = normalize_features(all_features)        
    centroids, cluster_labels = k_means_clustering(normalized_features, k)
    return cells, cluster_labels




# Perform the experiments with different parameter combinations and k values
parameter_combinations = [(8,1,16), (16,2,32)]  # Replace with your chosen parameter combinations
k_values = [3, 5]
for image_file in image_files:
    for binNumber, d, N in parameter_combinations:
        for k in k_values:   
        # Update the feature extraction functions with the new parameters
        # Run the main function and get the results
            image_path = os.path.join(images_path, image_file)
            cells, cluster_labels = mainfunction(binNumber, d, N, k)

            # Calculate the ratios and print them in a tabular format
            ratios = calculate_ratios(cells, cluster_labels, k)
            print(f"Ratios for binNumber={binNumber}, d={d}, N={N}, k={k}")
            print("Cluster | Inflammatory | Epithelial | Spindle-shaped")
            for i, (inf, epi, spi) in enumerate(ratios):
                print(f"{i+1}      | {inf:.2f}         | {epi:.2f}      | {spi:.2f}")

            # Visualize the clusters
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # Define your colors
            for image_file in image_files:
                image_path = os.path.join(images_path, image_file)
                result_img = visualize_clusters(image_path, cells, cluster_labels, k, colors)
                cv2.imshow(f"binNumber={binNumber}, d={d}, N={N}, k={k}, {image_file}", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
