import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv

# Function to draw landmarks on an image with indices
def draw_landmarks_on_image_with_index(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for face_idx, face_landmarks in enumerate(face_landmarks_list):
        for i, landmark in enumerate(face_landmarks):
            x, y, _ = landmark.x, landmark.y, landmark.z
            annotated_image = cv2.putText(annotated_image, str(i),
                                          (int(x * annotated_image.shape[1]), int(y * annotated_image.shape[0])),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated_image


# Function to save landmarks to CSV with indices
def save_landmarks_to_csv_with_index(face_landmarks_list, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['X', 'Y', 'Z'])  # Remove face_index and landmark_index columns
        for face_idx, face_landmarks in enumerate(face_landmarks_list):
            for landmark in face_landmarks:
                x, y, _ = landmark.x, landmark.y, landmark.z
                csvwriter.writerow([x, y, _])

# Function to save matrix to NumPy file
def save_matrix_to_np(matrix, output_path):
    np.save(output_path, matrix)

# Function to plot face blendshapes as a bar graph
def plot_face_blendshapes_bar_graph(face_blendshapes, output_path):
    if not face_blendshapes:
        print("No face blendshapes detected.")
        return

    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid potential conflicts


# Local file paths
model_file_path = '/home/aro/Schreibtisch/virtual_python/FER tasks/FacePrediction/new approach/face_landmarker.task'
image_folder = '/home/aro/Schreibtisch/AffectNet small/train_set/small_image_folder/left/images'

# Create FaceLandmarker object
base_options = python.BaseOptions(model_asset_path=model_file_path)
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Create output folders as siblings to the input image folder
parent_folder = os.path.dirname(image_folder)
blendshapes_folder = os.path.join(parent_folder, 'blendshapes')
transformation_folder = os.path.join(parent_folder, 'transformation_matrices')
visualization_folder = os.path.join(parent_folder, 'visualizations')
landmark_coordinates_folder = os.path.join(parent_folder, 'landmark_coordinates')

for folder in [blendshapes_folder, transformation_folder, visualization_folder, landmark_coordinates_folder]:
    os.makedirs(folder, exist_ok=True)

# Process each image in the folder
for image_filename in os.listdir(image_folder):
    if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_file_path = os.path.join(image_folder, image_filename)

        # Load the input image
        image = mp.Image.create_from_file(image_file_path)

        # Detect face landmarks from the input image
        detection_result = detector.detect(image)

        # Process the detection result and visualize it
        annotated_image = draw_landmarks_on_image_with_index(image.numpy_view(), detection_result)

        # Save annotated image with landmarks
        output_image_path = os.path.join(visualization_folder, f'{os.path.splitext(image_filename)[0]}_landmarks.jpg')
        cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # Save face blendshapes bar graph
        #output_graph_path = os.path.join(blendshapes_folder, f'{os.path.splitext(image_filename)[0]}_blendshapes_graph.png')
        #plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0], output_graph_path)

        # Save face landmarks coordinates
        output_landmarks_path = os.path.join(landmark_coordinates_folder, f'{os.path.splitext(image_filename)[0]}_landmarks.csv')
        save_landmarks_to_csv_with_index(detection_result.face_landmarks, output_landmarks_path)

        # Save facial transformation matrix as NumPy file
        output_matrix_path = os.path.join(transformation_folder, f'{os.path.splitext(image_filename)[0]}_transformation_matrix.npy')
        save_matrix_to_np(detection_result.facial_transformation_matrixes, output_matrix_path)
