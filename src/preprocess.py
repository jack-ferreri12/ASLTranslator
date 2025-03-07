import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set dataset paths
train_dir = "images/train"
test_dir = "images/test"

# Image settings
img_height = 224
img_width = 224
batch_size = 64  # Increased for stability


# Function to parse XML files and extract labels
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image filename
    filename = root.find("filename").text

    # Get class label (assuming single-label per image)
    object_tag = root.find("object")
    label = object_tag.find("name").text if object_tag is not None else "unknown"

    return filename, label


# Function to load images and labels
def load_dataset(image_folder):
    images = []
    labels = []
    class_names = set()  # Store unique class names

    for filename in os.listdir(image_folder):
        if filename.endswith(".xml"):  # Process XML files only
            xml_path = os.path.join(image_folder, filename)
            img_filename, label = parse_annotation(xml_path)

            # Convert label into an index
            class_names.add(label)

            # Load corresponding image
            img_path = os.path.join(image_folder, img_filename)
            if os.path.exists(img_path):  # Ensure image exists
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = cv2.resize(img, (img_width, img_height))  # Resize image

                images.append(img)
                labels.append(label)

    # Convert labels to indices
    class_names = sorted(list(class_names))
    labels = np.array([class_names.index(lbl) for lbl in labels], dtype=np.int32)

    return np.array(images), labels, class_names


# Load train and test datasets
train_images, train_labels, class_names = load_dataset(train_dir)
test_images, test_labels, _ = load_dataset(test_dir)

# Normalize images (TensorFlow expects float32 format)
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Convert to TensorFlow dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size).shuffle(len(train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

# Show some sample images with their labels
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_images[i])
    plt.title(class_names[train_labels[i]])
    plt.axis("off")
plt.show()
