from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
import threading
from PIL import ImageFont, ImageDraw, Image            ## 설치  ( pip install pillow ) pip install jamo
import numpy as np
from tensorflow.keras.models import load_model
import time


actions = ['지금', '까지', '3조', '발표', '들어주셔서', '감사합니다', '삭제']
seq_length = 30

model = load_model('./data/testmodel.h5')
# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

seq = []
action_seq = []
word = []

font = ImageFont.truetype('./static/fonts/SCDream6.otf', 20)


def home(request):
    context = {}

    return render(request, "home.html", context)

def DL(request):
    context = {}

    return render(request, "DL.html", context)

class VideoCamera(object):
    def __init__(self):
        
        self.img = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.img.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.img.release()


    def get_frame(self):
        ret, img = cap.read()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

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
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)

                draw.text(xy=(int(res.landmark[0].x*640), int(res.landmark[0].y*480 + 20)), text=action, font = font, fill=(255,255,255))
                img = np.array(img)
        content = ''
        for i in word:
            if i in content:
                pass
            else:
                content += i
                content += " "
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.text(xy=(10,30), text=content, font = font, fill=(255,255,255))
        img = np.array(img)

        _, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.img.read()
            
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def detectDL(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")
        pass
