import pickle
import os
import shutil
import numpy as np
from flask import Flask, render_template
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

app = Flask(__name__)

# Set the path to the dataset location (assuming "flower_images" folder in the cwd)
path = os.path.join(os.getcwd(), "flower_images/flower_images")

# Change the working directory to the path where the images are located
os.environ['OMP_NUM_THREADS'] = '1'

# Specify the filename for the pickled file
pickle_filename = "features.pickle"

# Construct the full path to the pickle file in the cwd
pickle_path = os.path.join(os.getcwd(), pickle_filename)

# Check if the pickle file exists
if os.path.exists(pickle_path):
    # Load data from the pickle file
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file)
    print("Data loaded from pickle file.")
else:
    try:
        # This list holds all the image filenames
        flowers = [file.name for file in os.scandir(path) if file.name.endswith('.png')]

        # VGG16 model
        model = VGG16()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)


        def extract_features(file, model):
            # Load the image as a 224x224 array
            img = load_img(file, target_size=(224, 224))
            # Convert from 'PIL.Image.Image' to a numpy array
            img = np.array(img)
            # Reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
            reshaped_img = img.reshape(1, 224, 224, 3)
            # Prepare image for the model
            imgx = preprocess_input(reshaped_img)
            # Get the feature vector
            features = model.predict(imgx, use_multiprocessing=True)
            return features


        data = {}

        # Loop through each image in the dataset
        for flower in flowers:
            # Try to extract the features and update the dictionary
            try:
                image_path = os.path.join(path, flower)
                feat = extract_features(image_path, model)
                data[flower] = feat
            except Exception as e:
                raise Exception(f"Error extracting features for {flower}: {e}")

        # Save the pickled data outside the loop
        with open(pickle_path, 'wb') as file:
            pickle.dump(data, file)
        print("Data saved to pickle file.")
    except Exception as e:
        raise Exception(f"Error processing images: {e}")

# Get a list of the filenames
filenames = np.array(list(data.keys()))

# Get a list of just the features
feat = np.array(list(data.values()))

# Reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1, 4096)

# Reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# Cluster feature vectors
kmeans = KMeans(n_clusters=5, n_init=10, random_state=22)
kmeans.fit(x)

# Holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
    # Limit the number of images to eight per cluster
    if len(groups[cluster]) < 8:
        groups[cluster].append(file)

# Create or check if cluster directories exist in the static folder
for cluster in range(5):
    cluster_path = os.path.join(os.getcwd(), 'static', 'flower_images', f'cluster_{cluster + 1}')
    print(f"Checking cluster directory: {cluster_path}")
    os.makedirs(cluster_path, exist_ok=True)

    # Check if images are present in the cluster directory, if not, transfer them
    cluster_files = os.listdir(cluster_path)
    if not cluster_files:
        print(f"Transferring images to cluster directory: {cluster_path}")
        cluster_data = {k: v for k, v in data.items() if k in filenames}  # Use the original filenames
        cluster_labels = kmeans.predict(x)  # Predict cluster labels
        cluster_data = {k: v for k, v in cluster_data.items() if cluster_labels[filenames == k][0] == cluster}
        filenames_to_copy = list(cluster_data.keys())[:8]  # Limit to eight filenames per cluster
        for file in filenames_to_copy:
            shutil.copy(os.path.join(path, file), cluster_path)

# Get a list of the filenames
filenames = np.array(list(data.keys()))

# Get a list of just the features
feat = np.array(list(data.values()))

# Reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1, 4096)

# Reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# Cluster feature vectors
kmeans = KMeans(n_clusters=5, n_init=10, random_state=22)
kmeans.fit(x)

cluster_clarity = {}
for i, center in enumerate(kmeans.cluster_centers_):
    cluster_points = x[kmeans.labels_ == i]
    distances = np.linalg.norm(cluster_points - center, axis=1)
    mean_distance = np.mean(distances)
    cluster_clarity[i] = mean_distance

# ... (previous code)

# Holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
    # Limit the number of images to eight per cluster
    if len(groups[cluster]) < 8:
        groups[cluster].append(file)

# Add logging statements
print("Groups before sorting:", groups)

# Sort clusters by index
sorted_clusters = sorted(list(groups.keys()))

# Add logging statements
print("Sorted clusters:", sorted_clusters)

# Holds the cluster id and the images { id: [images] }
groups_sorted = {}
for key in sorted_clusters:
    if key in groups:
        groups_sorted[key] = groups[key]

# Add logging statements
print("Groups after sorting:", groups_sorted)


# Route to index page
@app.route('/')
def index():
    return render_template('index.html', clusters=sorted_clusters, clustered_images=groups_sorted)
