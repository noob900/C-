import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to model input size
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame_preprocessed = preprocess_input(frame_rgb)  # Preprocess for MobileNet
    return np.expand_dims(frame_preprocessed, axis=0)  # Add batch dimension

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    input_frame = preprocess_frame(frame)

    # Perform prediction
    predictions = model.predict(input_frame)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Top 3 predictions

    # Display the predictions on the frame
    for i, (_, label, prob) in enumerate(decoded_predictions):
        text = f"{label}: {prob:.2f}"
        cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Camera Feed with Predictions', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()