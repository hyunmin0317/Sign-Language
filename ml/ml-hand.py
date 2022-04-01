import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

sum = []
font = ImageFont.truetype("SCDream6.otf", 25)
gesture = {
    0: "자세히", 1: "보아야", 2: "예쁘다", 3: "오래", 4: "사랑스럽다", 5: "너도", 6: "그렇다", 7: "지우기"
}
gesture_en = {
    0: "detail", 1: "see", 2: "beautiful", 3: "long", 4: "lovely", 5: "you", 6: "sodo", 7: "del"
}

startTime = time.time()
sentence = ''
file = np.genfromtxt('worddata.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()  ## K-NN 알고리즘 객체 생성
knn.train(angle, cv2.ml.ROW_SAMPLE, label)  ## train, 행 단위 샘플

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        joint = np.zeros((21, 3))  # 21개의 마디 부분 좌표 (x, y, z)를 joint에 저장
        for j, lm in enumerate(hand_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z]
        # 벡터 계산
        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
        v = v2 - v1

        # 벡터 길이 계산 (Normalize v)
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # arcos을 이용하여 15개의 angle 구하기
        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

        angle = np.degrees(angle)  # radian 값을 degree로 변경

        data = np.array([angle], dtype=np.float32)
        print(data)

        ret, results, neighbours, dist = knn.findNearest(data, 3)
        index = int(results[0][0])

        if index in gesture.keys():
            if time.time() - startTime > 3:
                startTime = time.time()
                # 다 지우기
                if index == 7:
                    sum.clear()
                elif index == 8:
                    # sentence = ''
                    # #sum.append(gesture[index])
                    sum.clear()
                else:
                    sum.append(gesture[index])  # 인식된 단어 리스트에 추가..
                startTime = time.time()

            cv2.putText(image, gesture_en[index].upper(), (
                int(hand_landmarks.landmark[0].x * image.shape[1] - 10), int(hand_landmarks.landmark[0].y * image.shape[0] + 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for i in sum:
            if i in sentence:
                pass
            else:
                sentence += " "
                sentence += i
                print(sentence)

        draw.text(xy=(20, 440), text=sentence, font=font, fill=(255, 255, 255))

        image = np.array(image)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()