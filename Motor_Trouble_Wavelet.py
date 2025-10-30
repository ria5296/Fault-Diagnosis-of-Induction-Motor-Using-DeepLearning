import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
import time

normal_data = pd.read_csv('정상상태_100개씩_100쌍.csv', encoding='utf-8')['value'].to_numpy()
rotor_fault_data = pd.read_csv('회전자고장_100개씩_100쌍.csv', encoding='utf-8')['value'].to_numpy()
bearing_fault_data = pd.read_csv('베어링고장_100개씩_100쌍.csv', encoding='utf-8')['value'].to_numpy()

def process_segments(data, segment_size=100):
    num_segments = len(data) // segment_size
    segments = []
    for i in range(num_segments):
        segment = data[i * segment_size:(i + 1) * segment_size]
        segments.append(segment)
    return segments

def wavelet_transform(data, sampling_period=0.01, scales=np.arange(1, 32), wavelet='cmor'):
    # data가 여러 세그먼트가 담긴 리스트일 경우 각 세그먼트에 대해 변환 후 리스트 반환
    magnitudes = []
    for segment in data:
        coefficients, frequencies = pywt.cwt(segment, scales, wavelet, sampling_period=sampling_period)
        magnitude = np.sum(np.abs(coefficients), axis=0)
        magnitudes.append(magnitude)
    return magnitudes

sampling_period = 0.01
scales = np.arange(1, 32)
wavelet = 'cmor'

# White Gaussian Noise 추가
noise = 0.1
normal_data = normal_data + np.random.normal(0, noise, normal_data.shape)
rotor_fault_data = rotor_fault_data + np.random.normal(0, noise, rotor_fault_data.shape)
bearing_fault_data = bearing_fault_data + np.random.normal(0, noise, bearing_fault_data.shape)

# Offset
offset = 0
normal_data += offset
rotor_fault_data += offset
bearing_fault_data += offset

Wavelet_Transform_start_time = time.time()

segment_size = 100
normal_segments = process_segments(normal_data, segment_size)
rotor_fault_segments = process_segments(rotor_fault_data, segment_size)
bearing_fault_segments = process_segments(bearing_fault_data, segment_size)

normal_magnitude = wavelet_transform(normal_segments, sampling_period, scales, wavelet)
rotor_fault_magnitude = wavelet_transform(rotor_fault_segments, sampling_period, scales, wavelet)
bearing_fault_magnitude = wavelet_transform(bearing_fault_segments, sampling_period, scales, wavelet)

Wavelet_Transform_end_time = time.time()

normal_magnitude = [(i , 0) for i in normal_magnitude]
rotor_fault_magnitude = [(i , 1) for i in rotor_fault_magnitude]
bearing_fault_magnitude = [(i , 2) for i in bearing_fault_magnitude]

normal_train_data, normal_test_data = train_test_split(normal_magnitude, test_size=0.3, random_state=42)
rotor_fault_train_data, rotor_fault_test_data = train_test_split(rotor_fault_magnitude, test_size=0.3, random_state=42)
bearing_fault_train_data, bearing_fault_test_data = train_test_split(bearing_fault_magnitude, test_size=0.3, random_state=42)

train_data = normal_train_data + rotor_fault_train_data + bearing_fault_train_data
train_data = pd.DataFrame(train_data, columns = ['value', 'target'])
test_data = normal_test_data + rotor_fault_test_data + bearing_fault_test_data
test_data = pd.DataFrame(test_data, columns = ['value', 'target'])

X_train = np.array(train_data['value'].tolist())
y_train = train_data['target'].values
X_test = np.array(test_data['value'].tolist())
y_test = test_data['target'].values

scaler = MinMaxScaler()
encoder = OneHotEncoder()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = encoder.transform(y_test.reshape(-1,1)).toarray()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape = (X_train.shape[1],)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=16)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

y_train_pred = model.predict(X_train)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)

Test_Start_time = time.time()
y_test_pred = model.predict(X_test)
Test_End_time = time.time()
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

Wavelet_Transform_time = Wavelet_Transform_end_time - Wavelet_Transform_start_time
Test_time = Test_End_time - Test_Start_time

Total_time = Wavelet_Transform_time + Test_time
print(f"총 소요 시간 : Wavelet 변환 + 테스트 시간 = {Wavelet_Transform_time:.4f}s + {Test_time:.4f}s = {Total_time:.4f}s")

train_cm = confusion_matrix(np.argmax(y_train, axis=1), y_train_pred_classes)
test_cm = confusion_matrix(np.argmax(y_test, axis=1), y_test_pred_classes)

import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title, accuracy):
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap='Blues')
    
    # 컬러바 크기 조정 (행렬에 맞게)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    total = np.sum(cm)
    correct = np.trace(cm)
    accuracy_percent = 100 * correct / total if total > 0 else 0

    # 상단 제목에 전체 정확도 표시
    plt.title(f"Total Accuracy: {accuracy_percent:.4f}%", fontsize=15)

    # 정확도 정보 텍스트 박스 추가 (조금 더 아래쪽에 위치 조정)
    ax.text(0.99, -0.13, f"Accuracy Count : {correct}/{total}", 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='black', alpha=0.8), color='white')

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(np.arange(cm.shape[1]))
    ax.set_yticklabels(np.arange(cm.shape[0]))

    # 셀마다 값 표기
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=10, bbox=dict(facecolor='none', edgecolor='none'))

    plt.tight_layout()
    plt.show()


plot_confusion_matrix(train_cm, "Training Confusion Matrix", test_accuracy)
plot_confusion_matrix(test_cm, "Testing Confusion Matrix", test_accuracy)
