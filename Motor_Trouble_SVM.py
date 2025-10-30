import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

# 데이터 로드
normal_data = pd.read_csv('정상상태_100개씩_100쌍.csv', encoding='utf-8')
bearing_fault_data = pd.read_csv('베어링고장_100개씩_100쌍.csv', encoding='utf-8')
rotor_fault_data = pd.read_csv('회전자고장_100개씩_100쌍.csv', encoding='utf-8')

data = pd.concat([normal_data, bearing_fault_data, rotor_fault_data], ignore_index=True)

# 특성과 레이블 추출
X = data[['value']]  # 100개의 입력 특성
y = data[['target']]  # 타겟 레이블

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

from sklearn.svm import SVC
model = SVC()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
model.score(X_scaled_train, y_train)

from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
print("훈련데이터 오차행렬 : \n", confusion_train)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

confusion_matrix_display = ConfusionMatrixDisplay(confusion_train)
confusion_matrix_display.plot()
plt.show()

from sklearn.metrics import classification_report
cfreport_train=classification_report(y_train, pred_train)
print("분류예측 레포트:\n", cfreport_train)

pred_test=model.predict(X_scaled_test)
model.score(X_scaled_test, y_test)

confusion_test=confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)

from sklearn.metrics import classification_report
cfreport_test=classification_report(y_test, pred_test)
print("분류예측 레포트:\n", cfreport_test)