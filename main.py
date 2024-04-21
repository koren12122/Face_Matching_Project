import itertools
import shutil
import numpy as np
from deepface import DeepFace
import os
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import matplotlib.pyplot as plt
from utils import single_exp, calculate_adjacency_matrix, \
    distance_matrix_statistics, graph_centrallity_clean, check_saved_graph, sanitize_column_names, plot_distribution
import pandas as pd
import lightgbm as lgb
import re
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import time
from itertools import combinations
import random
# from deepface.commons import distance as dst
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from deepface.modules.verification import find_euclidean_distance
from deepface.modules.verification import find_cosine_distance



def modify_and_return_df(df_name):
    df = pd.read_csv(df_name)
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    df = df.drop(columns=["file_x", "file_y"])
    df = df.drop(columns=["Unnamed0"])
    df.loc[df[df.decision == 'Yes'].index, 'decision'] = 1
    df.loc[df[df.decision == 'No'].index, 'decision'] = 0
    return df

def set_dataset_for_train():
    root_directory = os.getcwd() + "/face_dir/clusters/demo_dataset/"

    # List of models
    models = ['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace']

    # Dictionary to store embeddings
    all_embeddings = {}

    # Iterate over each folder in the root directory
    for person_folder in os.listdir(root_directory):
        person_folder_path = os.path.join(root_directory, person_folder)

        # Check if the item in the directory is a folder
        if os.path.isdir(person_folder_path):
            print(f"Processing images for {person_folder}:")

            # Initialize a dictionary for the current person
            person_embeddings = {}

            # Iterate over each image in the person's folder
            for image_name in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_name)

                # Initialize a dictionary for the current image
                image_embeddings = {}

                # Iterate over each model
                for model in models:
                    # Obtain embeddings using DeepFace
                    embeddings_obj = DeepFace.represent(
                        img_path=image_path,
                        model_name=model,
                        enforce_detection=False,
                        detector_backend='yolov8'
                    )

                    # Store the embeddings in the dictionary
                    image_embeddings[model] = embeddings_obj[0]['embedding']

                # Store the embeddings for the current image in the person's dictionary
                person_embeddings[image_name] = image_embeddings

            # Store the person's dictionary in the overall dictionary
            all_embeddings[person_folder] = person_embeddings

            print("------------------------------")

    for model in models:
        positive_pairs = []
        negative_pairs = []

        for person, embeddings in all_embeddings.items():
            # Create positive pairs (same person)
            person_images = list(embeddings.keys())
            if len(person_images) >= 2:
                positive_combinations = list(combinations(person_images, 2))
                positive_pairs.extend([(model, pair[0], pair[1]) for pair in positive_combinations])

            # Create negative pairs (different persons)
            other_people = [p for p in all_embeddings.keys() if p != person]
            for _ in range(len(person_images)):
                other_person = random.choice(other_people)
                other_person_images = list(all_embeddings[other_person].keys())
                if other_person_images:
                    random_image = random.choice(other_person_images)
                    negative_pairs.append((model, person_images[0], other_person, random_image))

        # Print or use the positive_pairs and negative_pairs as needed
        print(f"Model: {model}")
        print("Positive Pairs:", positive_pairs)
        print("Negative Pairs:", negative_pairs)
        print("Positive Pairs len:", len(positive_pairs))
        print("Negative Pairs len:", len(negative_pairs))
        print("------------------------------")






def f_beta(precision, recall, beta):
    beta_squared = beta ** 2
    numerator = (1 + beta_squared) * (precision * recall)
    denominator = (beta_squared * precision) + recall
    if denominator == 0:
        return 0  # Avoid division by zero
    return numerator / denominator

def train_xgboost(df_train, df_test, num_boost_round, threshold):
    tic = time.time()

    target_name = "decision"

    y_train = df_train[target_name].values
    x_train = df_train.drop(columns=[target_name]).values

    y_test = df_test[target_name].values
    x_test = df_test.drop(columns=[target_name]).values

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    params = {
        'objective': 'reg:squarederror', # reg:squarederror
        'eval_metric': 'rmsle',
        'is_unbalance': True,
        'num_leaves': 31
    }

    evallist = [(dtrain, 'train'), (dtest, 'test')]
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evallist, early_stopping_rounds=20) # obj=squared_log_func, feval=rmsle,

    predictions = bst.predict(dtest)
    print(predictions)
    prediction_classes = (predictions >= threshold).astype(int)

    y_test = y_test.astype(int)
    y_test = np.where(y_test > 0, 1, 0)

    cm = confusion_matrix(y_test, prediction_classes)
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    f_beta_score = f_beta(precision, recall, beta=0.15)

    print("----xgboost Test--------")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("F_beta (0.15): ", f_beta_score)

    toc = time.time()
    print("Total Training Time: ", round(toc - tic, 2))

    return f_beta_score





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

def finding_representations(directory):
    positives, positives_len = create_positive_dataset(directory)
    negatives = create_negative_dataset(directory, positives_len)
    # Concatenate the positive and negative pairs into one DataFrame
    df = pd.concat([positives, negatives]).reset_index(drop=True)

    instances = df[["file_x", "file_y"]].values.tolist()

    models = ['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace']
    metrics = ['cosine', 'euclidean_l2']

    image_dir = os.getcwd() + "/event_1_partition/images/"
    for model in models:
        for metric in metrics:
            distances = []
            for pair in instances:
                # assuming that pair[0] is the path to img1 and pair[1] is the path to img2
                resp_obj = DeepFace.verify(img1_path=image_dir + pair[0], img2_path=image_dir + pair[1], model_name=model,
                                           distance_metric=metric, detector_backend='yolov8')
                # Extract the distance and append it to the distances list
                distance = resp_obj[
                    "distance"]  # depending on the actual output structure, key might need to be adjusted
                print("inference time: ", resp_obj["time"])
                distances.append(distance)
            # After the innermost loop, add the distances to the dataframe
            df['%s_%s' % (model, metric)] = distances

    #save dataframe
    df.to_csv('df.csv')

def new_finding_representations(directory):
    positives, positives_len = create_positive_dataset(directory)
    negatives = create_negative_dataset(directory, positives_len)
    # Concatenate the positive and negative pairs into one DataFrame
    df = pd.concat([positives, negatives]).reset_index(drop=True)

    instances = df[["file_x", "file_y"]].values.tolist()


    models = ['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace']

    image_dir = os.getcwd() + "/event_1_partition/images/"
    for model in models:
        cosine_distances = []
        euclidean_l2_distances = []
        for pair in instances:
            # assuming that pair[0] is the path to img1 and pair[1] is the path to img2
            resp_obj = DeepFace.new_verify(img1_path=image_dir + pair[0], img2_path=image_dir + pair[1], model_name=model, detector_backend='yolov8')
            # Extract the distance and append it to the distances list
            cosine_distance = resp_obj["cosine_distance"]  # depending on the actual output structure, key might need to be adjusted
            euclidean_l2_distance = resp_obj["euclidean_l2_distance"]
            cosine_distances.append(cosine_distance)
            euclidean_l2_distances.append(euclidean_l2_distance)
            print("inference time: ", resp_obj["time"])
        # After the innermost loop, add the distances to the dataframe
        df['%s_cosine' % model] = cosine_distances
        df['%s_euclidean_l2' % model] = euclidean_l2_distances

    #save dataframe
    df.to_csv('df.csv')

def representations_dist_to_df(directory):
    models = ['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace']

    # Calculate representations for each model
    representations = calculate_representations(directory, models)

    # Create a DataFrame from the representations
    for model, data in representations.items():
        df = pd.DataFrame(data)
        df.to_csv(f'{model}_representations.csv', index=False)

    positives, positives_len = create_positive_dataset(directory)
    negatives = create_negative_dataset(directory, positives_len)
    # Concatenate the positive and negative pairs into one DataFrame
    df = pd.concat([positives, negatives]).reset_index(drop=True)

    instances = df[["file_x", "file_y"]].values.tolist()


    models = ['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace']

    image_dir = os.getcwd() + "/event_1_partition/images/"
    for model in models:
        cosine_distances = []
        euclidean_l2_distances = []
        for pair in instances:
            # assuming that pair[0] is the path to img1 and pair[1] is the path to img2
            resp_obj = DeepFace.new_verify(img1_path=image_dir + pair[0], img2_path=image_dir + pair[1], model_name=model, detector_backend='yolov8')
            # Extract the distance and append it to the distances list
            cosine_distance = resp_obj["cosine_distance"]  # depending on the actual output structure, key might need to be adjusted
            euclidean_l2_distance = resp_obj["euclidean_l2_distance"]
            cosine_distances.append(cosine_distance)
            euclidean_l2_distances.append(euclidean_l2_distance)
            print("inference time: ", resp_obj["time"])
        # After the innermost loop, add the distances to the dataframe
        df['%s_cosine' % model] = cosine_distances
        df['%s_euclidean_l2' % model] = euclidean_l2_distances

    #save dataframe
    df.to_csv('df.csv')


def calculate_representations(directory, models):
    representations = []
    image_files = os.listdir(directory)

    for model in models:
        embedding_dict = {}
        for img_file in image_files:
            img_path = os.path.join(directory, img_file)

            # Calculate the representation using DeepFace.represent
            embeddings_obj = DeepFace.represent(
                img_path=img_path,
                model_name=model,
                enforce_detection=False,
                detector_backend='yolov8'
            )

            for embedding in embeddings_obj:
                if tuple(embedding['embedding']) in embedding_dict:
                    embedding_dict[tuple(embedding['embedding'])].append(img_path)
                else:
                    embedding_dict[tuple(embedding['embedding'])] = [img_path]

        representations.append(embedding_dict)

    return representations

def old_get_embeddings(directory, model_name):
    face_list = []
    for root, dirs, files in os.walk(os.getcwd() + directory):
        for file in files:
            if ".jpg" in file:
                face_list.append(file)
    counter = 0
    embeddings = []
    for face in face_list:
        tic = time.time()
        embedding = DeepFace.represent(img_path=os.getcwd() + directory + face, model_name=model_name, enforce_detection=False, detector_backend='yolov8')
        embeddings.append(embedding[0]['embedding'])
        toc = time.time()
        print("inference time: ", round(toc - tic, 2))
        counter += 1
    return embeddings


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
    create_person_directories(clusters, embeddings_dict, "/face_dir/clusters/")
    print(len(clusters))
    exit()


def perform_mixup(df, alpha=0.5):

    def mixup(x1, x2, y1, y2, alpha):
        beta = np.random.beta(alpha, alpha)
        mixup_x = beta * x1 + (1 - beta) * x2
        mixup_y = beta * y1 + (1 - beta) * y2
        return mixup_x, mixup_y

    # Initialize lists to store mixed features and labels
    mixed_features = []
    mixed_labels = []

    # Perform mixup on pairs of samples
    for i in range(5 * len(df)):
        # Select two samples randomly
        index1 = np.random.randint(0, len(df))
        index2 = np.random.randint(0, len(df))
        x1, y1 = df.iloc[index1, 1:], df.iloc[index1]['decision']
        x2, y2 = df.iloc[index2, 1:], df.iloc[index2]['decision']
        mixed_feature, mixed_label = mixup(x1, x2, y1, y2, alpha)
        mixed_features.append(mixed_feature)
        mixed_labels.append(mixed_label)

    # Create a new DataFrame with mixed features and labels
    mixed_df = pd.DataFrame({
        'decision': mixed_labels,
        'VGG-Face_cosine': [f[0] for f in mixed_features],
        'VGG-Face_euclidean_l2': [f[1] for f in mixed_features],
        'Facenet_cosine': [f[2] for f in mixed_features],
        'Facenet_euclidean_l2': [f[3] for f in mixed_features],
        'Facenet512_cosine': [f[4] for f in mixed_features],
        'Facenet512_euclidean_l2': [f[5] for f in mixed_features],
        'ArcFace_cosine': [f[6] for f in mixed_features],
        'ArcFace_euclidean_l2': [f[7] for f in mixed_features]

    })

    return mixed_df



if __name__ == '__main__':
    df_train = pd.read_csv('training_set.csv')
    df_test = pd.read_csv('test_set.csv')

    train_xgboost(df_train, df_test, 360, 0.94)

    # max_score = 0
    # best_tree = None
    # best_threshold = None
    # for trees in [320, 360]:
    #     for threshold in [0.93, 0.94, 0.95]:
    #         score = train_xgboost(df, trees, threshold)
    #         if score > max_score:
    #             max_score = score
    #             best_tree = trees
    #             best_threshold = threshold
    #
    # print("Best Score Was: ", max_score)
    # print("Best Threshold Was: ", best_threshold)
    # print("Best Tree Was: ", best_tree)







    # train_lightgbm(df)

