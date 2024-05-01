import os
from deepface import DeepFace
import shutil
import time
from collections import defaultdict
import random
from sklearn.metrics.pairwise import cosine_similarity
from deepface.modules.verification import find_euclidean_distance
from deepface.modules.verification import find_cosine_distance
import pandas as pd
import re
import itertools
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

"""
This file is used to create the training dataset for the face recognition system.
Given a directory containing a collection of face images from event, the code in this file is responsible for splitting 
the images into clusters, where each cluster ideally corresponds to a unique individual.
 
Note that in order to create a high-quality dataset without noise, it is necessary to manually fix any clustering 
errors that may arise. This involves ensuring that:

    -Single Person per Cluster: Each cluster generated by the automated clustering algorithm corresponds to a single 
     unique individual, and does not contain face images of multiple different people.

    -Unique Clusters: There are no two separate clusters that both contain face images of the same person. 
     In other words, each person is represented by a single, distinct cluster in the dataset.
"""
def create_person_directories(clusters, embeddings_dict, output_dir):
    # Create output directory if it doesn't exist
    output_directory = os.getcwd() + output_dir
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through clusters
    for cluster_id, cluster in clusters.items():
        # Create a directory for the person
        person_directory = os.path.join(output_directory, f"Person_{cluster_id}")
        os.makedirs(person_directory, exist_ok=True)

        # Copy face images to the person's directory
        for embedding in cluster:
            file_name = embeddings_dict[embedding]
            source_path = os.path.join(os.getcwd(), "face_dir/" + file_name)
            destination_path = os.path.join(person_directory, file_name)
            shutil.copyfile(source_path, destination_path)

def get_embeddings(directory, model_name):
    embeddings_dict = {}
    for root, dirs, files in os.walk(os.getcwd() + directory):
        for file in files:
            if ".jpg" in file:
                tic = time.time()
                embedding = DeepFace.represent(img_path=os.getcwd() + directory + file, model_name=model_name, enforce_detection=False, detector_backend='yolov8')
                if len(embedding) == 1:
                    embeddings_dict[tuple(embedding[0]['embedding'])] = file
                else:
                    continue
                toc = time.time()
                print("Inference time for {}: {} seconds".format(file, round(toc - tic, 2)))
    return embeddings_dict


def split_to_clusters(embeddings_dict, threshold=0.4):
    clusters = defaultdict(list)

    # Compare each embedding with existing clusters
    for embedding, _ in embeddings_dict.items():
        assigned = False  # Flag to track if the embedding is assigned to any existing cluster

        # Shuffle the list of cluster IDs for random order
        cluster_ids = list(clusters.keys())
        random.shuffle(cluster_ids)
        print(cluster_ids)

        for cluster_id in cluster_ids:
            total_scores = 0
            counter = 0
            for cluster_embedding in clusters[cluster_id]:
                total_scores += cosine_similarity([embedding], [cluster_embedding])[0][0]
                counter += 1
            if counter > 0:
                average_score = total_scores / counter
            else:
                average_score = 0
            if average_score >= threshold:
                clusters[cluster_id].append(embedding)
                assigned = True
                break
        # If not assigned to any existing cluster, create a new cluster
        if not assigned:
            clusters[len(clusters)] = [embedding]

    return clusters

def face_clustering(dir):
    embeddings_dict = get_embeddings(dir, "VGG-Face")
    clusters = split_to_clusters(embeddings_dict, threshold=0.4)
    create_person_directories(clusters, embeddings_dict, "/face_dir/clusters/") # switch with your directory

def create_positive_dataset(directory):
    identities = {}
    positives = []

    # Iterate through each folder (identity) in the main directory
    for identity_folder in os.listdir(directory):
        identity_path = os.path.join(directory, identity_folder)

        # Check if it's a directory
        if os.path.isdir(identity_path):
            # List all files in the identity folder
            identity_files = os.listdir(identity_path)

            # Add the identity and corresponding files to the dictionary
            identities[identity_folder] = identity_files
            # Create positive pairs
            for i in range(0, len(identity_files) - 1):
                for j in range(i + 1, len(identity_files)):
                    positive = []
                    positive.append(identity_files[i])
                    positive.append(identity_files[j])
                    positives.append(positive)

    # Create a DataFrame from the positive pairs
    positives = pd.DataFrame(positives, columns=["file_x", "file_y"])
    positives["decision"] = "Yes"
    return positives, len(positives)

def create_negative_dataset(directory):
    identities = {}
    negatives = []

    # Iterate through each folder (identity) in the main directory
    for identity_folder in os.listdir(directory):
        identity_path = os.path.join(directory, identity_folder)

        # Check if it's a directory
        if os.path.isdir(identity_path):
            # List all files in the identity folder
            identities[identity_folder] = os.listdir(identity_path)

    samples_list = list(identities.values())

    # Generate negative pairs
    for i in range(0, len(identities) - 1):
        for j in range(i + 1, len(identities)):
            cross_product = itertools.product(samples_list[i], samples_list[j])
            cross_product = list(cross_product)

            for cross_sample in cross_product:
                negative = []
                negative.append(cross_sample[0])
                negative.append(cross_sample[1])
                negatives.append(negative)
                negatives.append(negative)

    # Create a DataFrame from the negative pairs and sample the same number as positives
    negatives = pd.DataFrame(negatives, columns=["file_x", "file_y"])
    negatives["decision"] = "No"
    # negatives_df = negatives.sample(positives_df_len)
    return negatives


def cosine_func(emb1, emb2):
    # return dst.findCosineDistance(emb1, emb2)
    return find_cosine_distance(emb1, emb2)

def euclidean_func(emb1, emb2):
    # return dst.findEuclideanDistance(
    #                 dst.l2_normalize(emb1), dst.l2_normalize(emb2))
    return find_euclidean_distance(emb1, emb2)

def create_the_csv_file(directory, df_name = 'new_df.csv'):
    # Assuming create_positive_dataset and create_negative_dataset are defined elsewhere
    positives, positives_len = create_positive_dataset(directory)
    negatives = create_negative_dataset(directory)

    # Concatenate the positive and negative pairs into one DataFrame
    df = pd.concat([positives, negatives]).reset_index(drop=True)

    instances = df[["file_x", "file_y"]].values.tolist()
    models = ['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace']
    metrics = {'cosine': cosine_func, 'euclidean_l2': euclidean_func}

    image_dir = os.getcwd() + "/face_dir/"
    embeddings_cache = {}

    for model in models:
        # Cache embeddings for each image to avoid recalculating
        embeddings_cache[model] = {}
        for img_file in set(df['file_x']).union(set(df['file_y'])):
            img_path = image_dir + img_file
            embeddings = DeepFace.represent(
                img_path=img_path,
                model_name=model,
                enforce_detection=False,
                detector_backend='yolov8'
            )

            embeddings_cache[model][img_file] = embeddings[0]['embedding']
        # Calculate distances
        for metric_name, metric_func in metrics.items():
            distances = []
            for pair in instances:
                emb1 = embeddings_cache[model][pair[0]]
                emb2 = embeddings_cache[model][pair[1]]
                distance = metric_func(emb1, emb2)
                distances.append(distance)
            # After the innermost loop, add the distances to the dataframe
            df['%s_%s' % (model, metric_name)] = distances

    # Save DataFrame to CSV
    df.to_csv(df_name)

def modify_and_return_df(df_name):
    df = pd.read_csv(df_name)
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    df = df.drop(columns=["file_x", "file_y"])
    df = df.drop(columns=["Unnamed0"])
    df.loc[df[df.decision == 'Yes'].index, 'decision'] = 1
    df.loc[df[df.decision == 'No'].index, 'decision'] = 0
    return df



images_path = "enter/your/path/to/images/"
face_clustering(images_path)
face_directory = images_path + 'face_dir/clusters/'  # The face clusters (change if needed)
df_name = 'enter_df_name.csv'
create_the_csv_file(os.getcwd() + face_directory, df_name)
df = modify_and_return_df(df_name)
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)
df_train.to_csv('training_set.csv', index=False)
df_test.to_csv('test_set.csv', index=False)

