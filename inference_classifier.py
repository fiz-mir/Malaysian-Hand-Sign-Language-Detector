import time
import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O',
    14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

sentence = ""
existing_alphabets = set()
cooldown_time = 2  # Set the cooldown time in seconds
last_prediction_time = time.time() - cooldown_time  # Initialize the last prediction time

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            handedness = results.multi_handedness[idx]
            hand_label = handedness.classification[0].label
            hand_score = handedness.classification[0].score

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

                # Display accuracy for existing alphabets
                if labels_dict[i] in existing_alphabets:
                    text = f'{labels_dict[i]}: {hand_score:.2f}'
                    text_position = (10, 30 + i * 30)

                    # Check if text position is within screen boundaries
                    if 0 <= text_position[0] < W and 0 <= text_position[1] < H:
                        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Check cooldown
        if time.time() - last_prediction_time >= cooldown_time:
            # Double the features to match the expected input
            prediction = model.predict([np.asarray(data_aux * 2)])
            predicted_character = labels_dict[int(prediction[0])]
            sentence += predicted_character
            existing_alphabets.add(predicted_character)
            last_prediction_time = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, sentence, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)

    # Check for key press 'r' to reset
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        sentence = ""
        existing_alphabets = set()

cap.release()
cv2.destroyAllWindows()
