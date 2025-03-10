import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (255, 0, 0)  # red


# Function to visualize detection results
def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the detected objects and returns the image."""
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_x, start_y = bbox.origin_x, bbox.origin_y
        end_x, end_y = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

  # Crop the detected object with margin
        cropped_img = image[max(0, start_y - MARGIN): min(image.shape[0], end_y + MARGIN),
                      max(0, start_x - MARGIN): min(image.shape[1], end_x + MARGIN)]

  # Resize cropped image
        cropped_img = cv2.resize(cropped_img, (300, 300))

   # Draw bounding box only around detected object
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), TEXT_COLOR, 3)

 # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (start_x, max(20, start_y - 5))
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image


# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create an ObjectDetector object
base_options = python.BaseOptions(model_asset_path='model_fp16.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

  # Convert OpenCV image to MediaPipe format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

 # Detect objects in the frame
    detection_result = detector.detect(mp_image)

 # Process and visualize the detection result
    annotated_frame = visualize(frame, detection_result)

  # Display the result
    cv2.imshow('Detected Objects', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
