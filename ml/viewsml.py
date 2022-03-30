from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
import threading
from PIL import ImageFont, ImageDraw, Image            ## 설치  ( pip install pillow )
import numpy as np
import time


sum = []

font = ImageFont.truetype("SCDream6.otf", 25)
        

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

gesture = {
    0:"자세히",1:"보아야",2:"예쁘다",3:"오래",4:"사랑스럽다",5:"너도",6:"그렇다",7:"지우기"
}
gesture_en = {
    0:"detail", 1:"see", 2:"beautiful",3:"long",4:"lovely",5:"you",6:"sodo",7:"del"
}

## 최대 1개의 손만 인식
max_num_hands = 2
hands = mp_hands.Hands(max_num_hands = max_num_hands,
                    min_detection_confidence = 0.7,
                    min_tracking_confidence = 0.7)

file = np.genfromtxt('worddata.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

knn = cv2.ml.KNearest_create()              ## K-NN 알고리즘 객체 생성
knn.train(angle, cv2.ml.ROW_SAMPLE, label)  ## train, 행 단위 샘플

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        self.startTime = time.time()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()
    

    def get_frame(self):        
       
        flag = []
        
        prev_index = 0
        sentence = ''
        recognizeDelay = 0.1    

        image = self.frame
        video = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video = cv2.flip(video, 1)

        video.flags.writeable = False

        result = hands.process(video)

        video.flags.writeable = True
        video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)

        # if result.multi_hand_landmarks:
        #     for res in result.multi_hand_landmarks:
        #         joint = np.zeros((21, 3))
        #         for j,lm in enumerate(res.landmark):
        #             joint[j] = [lm.x, lm.y, lm.z]
        if result.multi_hand_landmarks is not None:      # 손 인식 했을 경우
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))                 # 21개의 마디 부분 좌표 (x, y, z)를 joint에 저장
                for j,lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]
                # 벡터 계산
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],:]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],:]
                v = v2 - v1

                # 벡터 길이 계산 (Normalize v)
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # arcos을 이용하여 15개의 angle 구하기
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18],:],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],:]))

                angle = np.degrees(angle)  # radian 값을 degree로 변경

                data = np.array([angle], dtype=np.float32)

                ret, results,neighbours,dist = knn.findNearest(data,3)
                index = int(results[0][0])

                if index in gesture.keys():
                    if time.time() - self.startTime > 3:
                        self.startTime = time.time()
                                # 다 지우기
                        if index == 7:
                            sum.clear()
                        elif index == 8:
                            # sentence = ''
                            # #sum.append(gesture[index])
                            sum.clear()
                        else:
                            sum.append(gesture[index]) #인식된 단어 리스트에 추가..
                        startTime = time.time()

                    cv2.putText(video,gesture_en[index].upper(),(int(res.landmark[0].x * video.shape[1] - 10),int(res.landmark[0].y * video.shape[0] + 40)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),3)

                mp_drawing.draw_landmarks(video,res,mp_hands.HAND_CONNECTIONS)

        video = Image.fromarray(video)
        draw = ImageDraw.Draw(video)
        for i in sum:
            if i in sentence:
                pass
            else:
                sentence += " "
                sentence += i

        draw.text(xy=(20, 440), text = sentence, font=font, fill=(255, 255, 255))
        
        video = np.array(video)
        
        _, jpeg = cv2.imencode('.jpg', video)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


cam = VideoCamera()