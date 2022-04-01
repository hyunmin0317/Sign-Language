import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

##########데이터 로드

df = pd.read_csv('hands.csv')

##########데이터 분석

##########데이터 전처리

x_data = df.drop(['W'], axis=1)
y_data = df['W']

print(x_data.head())
'''
    SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
28            5.2           3.4            1.4           0.2
91            6.1           3.0            4.6           1.4
41            4.5           2.3            1.3           0.3
45            4.8           3.0            1.4           0.3
57            4.9           2.4            3.3           1.0
'''

le = LabelEncoder()
le.fit(y_data)
print(le.classes_) #['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
#labels = le.classes_
# labels = ['세토사', '버시칼라', '버지니카']
y_data = le.transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=10000, stratify=y_data)

##########모델 생성

model = LogisticRegression(max_iter=10000)

##########모델 학습

model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_train, y_train)) #

print(model.score(x_test, y_test)) #0.9777777777777777

# 모델 저장
joblib.dump(model,'./Logistic.pkl')
##########모델 예측

# x_test = np.array([
#     [5.3, 3.7, 1.5, 0.2]
# ])
#
# y_predict = model.predict(x_test)
# label = labels[y_predict[0]]
# y_predict = model.predict_proba(x_test)
# confidence = y_predict[0][y_predict[0].argmax()]
#
# print(label, confidence) #