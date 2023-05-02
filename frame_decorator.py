import cv2
import numpy as np
import threading
import queue
import random as r


# Define emotions and their corresponding angles in radians
emotions = {
    "Angry": 0,
    "Disgust": 2 * np.pi / 7,
    "Fear": 4 * np.pi / 7,
    "Happy": 6 * np.pi / 7,
    "Neutral": 8 * np.pi / 7,
    "Sad": 10 * np.pi / 7,
    "Surprise": 12 * np.pi / 7,
}


def get_emotion_percentages_from_ml_model(frame):
    # Implement your ML model inference code here, using the 'frame' input.
    # For example purposes, we'll return random percentages for each emotion.
    percentages = {emotion: np.random.random() for emotion in emotions}
    total = sum(percentages.values())
    return {emotion: percentage / total for emotion, percentage in percentages.items()}


def process_frames():
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            emotion_percentages = get_emotion_percentages_from_ml_model(frame)
            frame_queue.put((emotion_percentages, frame))


def draw_arrow(image, angle, color=(255, 255, 255)):
    center = (320, 240)
    radius = 200
    endpoint = (
        int(center[0] + radius * np.cos(angle)),
        int(center[1] + radius * np.sin(angle)),  # Change - to +
    )
    cv2.arrowedLine(image, center, endpoint, color, 2)


def draw_circle_frame(image, center, radius, color=(255, 255, 255)):
    cv2.circle(image, center, radius, color, 2)


def draw_emotion_labels(image, emotion_percentages):
    center = (320, 240)
    radius = 200
    for idx, emotion in enumerate(emotions):
        angle = emotions[emotion]
        label_position = (
            int(center[0] + radius * np.cos(angle)),
            int(center[1] - radius * np.sin(angle)),
        )
        percentage_text = (
            f"{emotion.capitalize()} ({emotion_percentages[emotion] * 100:.1f}%)"
        )
        cv2.putText(
            image,
            percentage_text,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def create_circular_mask(frame, center, radius):
    # Create a mask with the same shape as the frame
    mask = np.zeros_like(frame)

    # Draw a white circle at the center of the mask
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    # Draw a black circle at the center of the mask with a smaller radius to make the center transparent
    cv2.circle(mask, center, int(radius / 2), (0, 0, 0), -1)

    # Convert the mask to a single-channel image
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Invert the mask so that the transparent region is white
    mask = cv2.bitwise_not(mask)

    # Convert the mask to a 3-channel image with transparency
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)

    return mask


def create_circle_mask(frame, center, radius):
    mask = np.zeros_like(frame)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    return cv2.addWeighted(frame, 0.5, mask, 0.3, 0)


def plot_emotions_circle(
    emotion_values, frame, emotions, radius=200, center=(320, 240)
):
    # Define the emotions and their colors
    # colors = [(0, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0)]
    colors = [
        (r.randrange(0, 255), r.randrange(0, 255), r.randrange(0, 255))
        for _ in range(len(emotions))
    ]

    # Create a black background image
    background = np.zeros_like(frame)

    # Draw the circle
    cv2.circle(background, center, radius, (255, 255, 255), -1)

    # Draw the emotion labels
    label_padding = int(radius / 4)
    angles = np.linspace(0, 360, len(emotions) + 1)[:-1]
    for i, angle in enumerate(angles):
        x = int(center[0] + (radius + label_padding) * np.cos(np.radians(angle)))
        y = int(center[1] + (radius + label_padding) * np.sin(np.radians(angle)))
        text_size = cv2.getTextSize(emotions[i], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = int(x - text_size[0] / 2)
        text_y = int(y + text_size[1] / 2)
        cv2.putText(
            background,
            emotions[i],
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colors[i],
            2,
            cv2.LINE_AA,
        )

    # Get the emotion with the highest percentage and calculate its position on the circle
    max_emotion_index = np.argmax(emotion_values)
    bias = 0.2  # Bias towards the max emotion
    max_angle = angles[max_emotion_index] + (360 / len(emotions) * bias)

    # Calculate the position of the dot
    dot_angle = max_angle + 360 * sum(emotion_values[:max_emotion_index]) / sum(
        emotion_values
    )
    dot_position = (
        int(center[0] + radius * np.cos(np.radians(dot_angle))),
        int(center[1] + radius * np.sin(np.radians(dot_angle))),
    )

    # Draw the dot
    dot_radius = int(radius / 20)
    cv2.circle(background, dot_position, dot_radius, colors[max_emotion_index], -1)

    # Apply the mask to the frame
    mask = create_circle_mask(frame, center, radius)
    masked_frame = cv2.bitwise_and(frame, mask)

    # Add the masked frame to the background image
    output = cv2.add(background, masked_frame)

    # Add padding to the output image
    border_thickness = int(radius / 10)
    border_color = (255, 255, 255)
    output = cv2.copyMakeBorder(
        output,
        border_thickness,
        border_thickness,
        border_thickness,
        border_thickness,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )

    return output


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
    else:
        frame_queue = queue.Queue()

        # Start the thread for processing frames
        processing_thread = threading.Thread(target=process_frames, daemon=True)
        processing_thread.start()

        while True:
            try:
                emotion_percentages, frame = frame_queue.get(timeout=1)
                max_emotion = max(emotion_percentages, key=emotion_percentages.get)
                angle = emotions[max_emotion]
                frame = create_circle_mask(frame, (320, 240), 200)
                draw_arrow(frame, angle)
                draw_circle_frame(frame, (320, 240), 200)
                draw_emotion_labels(frame, emotion_percentages)
                cv2.imshow("Emotion Circle", frame)
            except queue.Empty:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
