import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("asl_translator_recovered.keras")

# Define image dimensions (same as training)
img_height = 224
img_width = 224

# Define class labels (Update this to match your training labels)
class_names = ["hello", "yes", "no", "thanks", "iloveyou"]  # Adjust as needed

# Open webcam
cap = cv2.VideoCapture(0)

# Get video frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the size and position of the box (bigger & centered)
box_size = 500  # Increased size to fully capture two-handed signs
box_x1 = (frame_width // 2) - (box_size // 2)
box_y1 = (frame_height // 2) - (box_size // 2)
box_x2 = box_x1 + box_size
box_y2 = box_y1 + box_size

# Variables for motion detection
prev_frame = None
motion_threshold = 500000  # Adjust as needed

# Variables for smoothing predictions
prediction_history = []
history_length = 5  # Average over last 5 frames

print("📸 Starting ASL Translator... Keep your hand inside the box. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute frame difference for motion detection
    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, gray)
        motion = np.sum(frame_diff)
    else:
        motion = 0

    prev_frame = gray  # Update previous frame

    # Draw a thicker box in the middle of the screen
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 4)  # Thicker border

    # Extract only the region inside the box
    hand_region = frame[box_y1:box_y2, box_x1:box_x2]

    # Preprocess the hand region
    img = cv2.resize(hand_region, (img_width, img_height))  # Resize to model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img.astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make a prediction only if motion is detected (useful for signs like "Thanks")
    if motion > motion_threshold:
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)  # Get class index
        confidence = np.max(predictions)  # Get confidence score

        # Only add to history if confidence is high
        if confidence > 0.65:
            prediction_history.append(predicted_class)
            if len(prediction_history) > history_length:
                prediction_history.pop(0)  # Keep history length fixed

            # Get the most common prediction over the last few frames
            most_common_prediction = max(set(prediction_history), key=prediction_history.count)

            label = f"{class_names[most_common_prediction]} ({confidence:.2f})"
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show webcam feed with the tracking box
    cv2.imshow("ASL Translator - Keep Your Hand Inside the Box", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ ASL Translator Stopped.")
