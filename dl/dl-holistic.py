import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from tensorflow.python.keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

actions = ['지금', '까지', '3조', '발표', '들어주셔서', '감사합니다', '삭제']
seq_length = 30
model = load_model('testmodel.h5')

seq = []
action_seq = []
word = []
font = ImageFont.truetype('SCDream6.otf', 20)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
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
    results = holistic.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.right_hand_landmarks:
      for hand_landmarks in results.right_hand_landmarks:
        joint = np.zeros((21, 4))  # 21개의 마디 부분 좌표 (x, y, z)를 joint에 저장
        for j, lm in enumerate(results.right_hand_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
        # 벡터 계산
        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
        v = v2 - v1

        # 벡터 길이 계산 (Normalize v)
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # arcos을 이용하여 15개의 angle 구하기
        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

        angle = np.degrees(angle)  # radian 값을 degree로 변경

        d = np.concatenate([joint.flatten(), angle])
        seq.append(d)

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        if len(seq) < seq_length:
            continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
        # 인퍼런스 한 결과를 뽑아낸다
        y_pred = model.predict(input_data).squeeze()
        # 어떠한 인덱스 인지 뽑아낸다
        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]
        # confidence 가 90%이하이면 액션을 취하지 않았다 판단
        if conf < 0.9:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 7:
            continue

        # 마지막 3번 반복되었을 때 진짜로 판단
        #this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            #this_action = action

            if action == '삭제':
                word.clear()
            else:
                word.append(action)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        draw.text(xy=(int(hand_landmarks.landmark[0].x*640), int(hand_landmarks.landmark[0].y*480 + 20)), text=action, font = font, fill=(255,255,255))
        image = np.array(image)
        content = ''
        for i in word:
            if i in content:
                pass
            else:
                content += i
                content += " "
                print(content)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text(xy=(10,30), text=content, font = font, fill=(255,255,255))
        image = np.array(image)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()