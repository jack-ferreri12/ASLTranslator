import tensorflow as tf
import cv2
import numpy as np
import os

model = tf.keras.models.load_model("asl_translator_recovered.keras")

class_names = ["hello", "yes", "no", "thanks", "iloveyou"]

test_images = ["test_yes.jpg", "test_thankyou.jpg"]

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"❌ Image {img_path} not found!")
        continue

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    print(f"🔍 {img_path}: Predicted → {class_names[predicted_class]} ({confidence:.2f})")
