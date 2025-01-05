import cv2
from ultralytics import YOLO
import os
import time

def main():
    # Path to the exported NCNN model
    model_path = "best_ncnn_model"

    # Paths to save the images
    base_dir = "images"
    original_dir = os.path.join(base_dir, "original")
    cropped_dir = os.path.join(base_dir, "cropped")
    os.makedirs(original_dir, exist_ok=True)  # Ensure the original directory exists
    os.makedirs(cropped_dir, exist_ok=True)  # Ensure the cropped directory exists

    # Initialize the YOLO model
    ncnn_model = YOLO(model_path, task="detect")

    # Open the camera stream
    cap = cv2.VideoCapture(0)  # Use the default camera
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Synchronization interval in seconds
    interval_between_inference = 0.35

    while True:
        # Record the start time for interval tracking
        start_time = time.time()

        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        # Run inference directly on the captured frame
        results = ncnn_model(frame)

        # Generate a timestamp for the filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save the original frame resized to 224x224
        resized_original = cv2.resize(frame, (224, 224))
        original_image_path = os.path.join(original_dir, f"original_{timestamp}.jpg")
        cv2.imwrite(original_image_path, resized_original)
        print(f"Original image saved to: {original_image_path}")

        # Variables to store the bounding box with the highest confidence score
        best_box = None
        best_score = 0

        # Find the bounding box with the highest score
        for result in results:
            boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates [x_min, y_min, x_max, y_max]
            scores = result.boxes.conf.numpy()  # Confidence scores

            for i, box in enumerate(boxes):
                score = scores[i]

                # Update if the current box has a higher score
                if score > best_score:
                    best_box = box
                    best_score = score

        # Crop the image to the highest confidence bounding box
        if best_box is not None:
            x_min, y_min, x_max, y_max = map(int, best_box)  # Convert to integers

            # Ensure bounding box is within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            # Crop the image to the bounding box
            cropped_image = frame[y_min:y_max, x_min:x_max]

            # Check the size of the cropped image
            if cropped_image.shape[0] < 224 or cropped_image.shape[1] < 224:
                cropped_image_path = os.path.join(cropped_dir, f"cropped_{timestamp}.jpg")
                cv2.imwrite(cropped_image_path, cropped_image)
                print(f"Cropped image saved to: {cropped_image_path}")
            else:
                # Resize the cropped image to 224x224 and save it
                resized_cropped = cv2.resize(cropped_image, (224, 224))
                cropped_image_path = os.path.join(cropped_dir, f"cropped_{timestamp}.jpg")
                cv2.imwrite(cropped_image_path, resized_cropped)
                print(f"Cropped image saved to: {cropped_image_path}")
        else:
            print("No bounding box detected.")

        # Calculate the time spent in processing
        elapsed_time = time.time() - start_time

        # Sleep for the remaining time in the interval
        sleep_time = max(0, interval_between_inference - elapsed_time)
        time.sleep(sleep_time)

        # Exit the loop when 'q' is pressed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("Exiting the script...")
            break
        elif key == 27:  # Escape key
            print("Escape key pressed. Exiting the script...")
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
