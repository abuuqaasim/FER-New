import os
import time
import numpy as np
from keras import layers, models, Input, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Function to load landmarks from folder
def load_landmarks_from_folder(folder_path):
    landmarks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                landmark_data = [float(coord) for coord in file.read().split()]
                landmarks.append(landmark_data)
    return np.array(landmarks)

# Load left and right landmarks
left_landmarks = load_landmarks_from_folder('/home/aro/Schreibtisch/virtual_python/FER-New/FacePrediction2/new approach/MVC-002F/landmark_coordinates')
right_landmarks = load_landmarks_from_folder('/home/aro/Schreibtisch/virtual_python/FER-New/FacePrediction2/new approach/MVC-004F/landmark_coordinates')

# Assuming you have the same number of images for both sides
num_images = len(left_landmarks)

# Flatten the landmark arrays to (num_images, num_landmarks * 3)
x_left = np.array(left_landmarks).reshape((num_images, -1))
x_right = np.array(right_landmarks).reshape((num_images, -1))

# Split the data into training and validation sets
x_left_train, x_left_test, x_right_train, x_right_test = train_test_split(x_left, x_right, test_size=0.37, random_state=42)

# Concatenate the training sets
combined_x_left_train = np.concatenate((x_left_train, x_right_train), axis=0)  #training inputs
combined_x_right_train = np.concatenate((x_right_train, x_left_train), axis=0) #training outputs

# Define your model
input_shape = x_left.shape[1]
img_input = Input(shape=(input_shape,))

hidden1 = layers.Dense(100, activation='relu', kernel_initializer='he_normal')(img_input)
hidden2 = layers.Dense(100, activation='relu', kernel_initializer='he_normal')(hidden1)

# Output layer for predicting right from left
out_right = layers.Dense(input_shape, activation='linear', name='out_right')(hidden2)

# Output layer for predicting left from right
out_left = layers.Dense(input_shape, activation='linear', name='out_left')(hidden2)

model = Model(inputs=img_input, outputs=[out_right, out_left])

# Compile the model
model.compile(optimizer='adam', loss={'out_right': 'mse', 'out_left': 'mse'})

# Measure training time
start_time_training = time.time()
model.fit(combined_x_left_train, {'out_right': combined_x_right_train, 'out_left': combined_x_left_train}, epochs=150, batch_size=32, verbose=2)
end_time_training = time.time()
training_time = end_time_training - start_time_training
print(f"Training Time: {training_time:.2f} seconds")

# Measure inference time
start_time_inference = time.time()
x_hat_right, x_hat_left = model.predict(x_left_test)
end_time_inference = time.time()
inference_time = end_time_inference - start_time_inference
print(f"Inference Time: {inference_time:.2f} seconds")

# Calculate the accuracy
left_to_right_error = mean_squared_error(x_right_test, x_hat_right)
right_to_left_error = mean_squared_error(x_left_test, x_hat_left)

print(f'Mean Squared Error on test Set L2R: {left_to_right_error}')
print(f'Mean Squared Error on test Set R2L: {right_to_left_error}')



# Function to visualize samples with indices as markers
def visualize_samples(actual, predicted, num_samples=10):
    sample_indices = np.random.choice(len(actual), num_samples, replace=False)

    for i, idx in enumerate(sample_indices):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(actual[idx, 0::3], -actual[idx, 1::3], label='Actual', marker='o', c='blue')
        for j, (x, y) in enumerate(zip(actual[idx, 0::3], -actual[idx, 1::3])):
            plt.text(x, y, str(j), fontsize=8, ha='right', va='bottom', color='blue')  # Display index as text
        plt.title('Actual Landmarks')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(predicted[idx, 0::3], -predicted[idx, 1::3], label='Predicted', marker='x', c='red')
        for j, (x, y) in enumerate(zip(predicted[idx, 0::3], -predicted[idx, 1::3])):
            plt.text(x, y, str(j), fontsize=8, ha='right', va='bottom', color='red')  # Display index as text
        plt.title('Predicted Landmarks')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Visualize ten samples with indices as markers
visualize_samples(x_left_test, x_hat_left)
