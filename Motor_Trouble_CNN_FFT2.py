import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.fft import fft, fftfreq
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import time

# 데이터 불러오기
normal_data = pd.read_csv('정상상태_100개씩_100쌍.csv', encoding='utf-8')
bearing_fault_data = pd.read_csv('베어링고장_100개씩_100쌍.csv', encoding='utf-8')
rotor_fault_data = pd.read_csv('회전자고장_100개씩_100쌍.csv', encoding='utf-8')

data = pd.concat([normal_data, bearing_fault_data, rotor_fault_data], ignore_index=True)

def perform_fft(value_data):
    t = np.linspace(0, 100, 10000, endpoint=False)  # 시간 축 생성
    signal = np.zeros(10000)  # 빈 신호 배열
    
    # 주파수에 따라 사인파 생성
    for frequency in value_data:
        signal += np.sin(2 * np.pi * frequency * t) 
        
        if noise_control == True: # 직류성분 노이즈 추가 부분
            bias = 1.0
            signal = signal + bias
            
    signal_fft = np.abs(fft(signal))
    signal_fft[0] = 0  # DC 성분 제거
    frequency_bins = fftfreq(len(signal_fft), d=1/10000)  # 주파수 축 생성
    half_length = len(signal_fft) // 2  # 양의 주파수 부분만 사용

    # 진폭 기준으로 정렬하여 상위 5개 주파수 선택
    top_5_frequencies = sorted(range(half_length), key=lambda i: signal_fft[i], reverse=True)[:5]
    
    return [frequency_bins[i] for i in top_5_frequencies]

def process_segments(data, segment_size=100):
    global noise_control
    noise_control = False
    
    if len(data) == 3000:
        noise_control = True
        
    num_segments = len(data) // segment_size
    segments = []
    for i in range(num_segments):
        segment = data.iloc[i * segment_size:(i + 1) * segment_size]['value']  # 'value' 열만 추출
        fft_result = perform_fft(segment)
        segments.append({'fft_result': fft_result})  # 딕셔너리 형태로 저장
    return segments 

# 훈련 및 테스트 데이터 분할
normal_train_data, normal_test_data = train_test_split(normal_data, test_size=0.3, random_state=42)
bearing_fault_train_data, bearing_fault_test_data = train_test_split(bearing_fault_data, test_size=0.3, random_state=42)
rotor_fault_train_data, rotor_fault_test_data = train_test_split(rotor_fault_data, test_size=0.3, random_state=42)

# 훈련 데이터 처리
normal_train_peak_amplitudes = process_segments(normal_train_data)
bearing_fault_train_peak_amplitudes = process_segments(bearing_fault_train_data)
rotor_fault_train_peak_amplitudes = process_segments(rotor_fault_train_data)

# 결과를 (FFT 결과, 레이블) 형태로 정리
normal_train_100 = [(i['fft_result'], 0) for i in normal_train_peak_amplitudes]
bearing_fault_train_100 = [(i['fft_result'], 1) for i in bearing_fault_train_peak_amplitudes]
rotor_fault_train_100 = [(i['fft_result'], 2) for i in rotor_fault_train_peak_amplitudes]

data_train_100 = normal_train_100 + bearing_fault_train_100 + rotor_fault_train_100
data_train = pd.DataFrame(data_train_100, columns=['fft_result', 'target'])

# FFT 결과에서 주파수만 추출하여 'value' 열 생성
data_train['value'] = data_train['fft_result']

# 데이터 섞기
data_train = data_train.sample(frac=1, random_state=42).reset_index(drop=True)

# X_train 및 y_train 정의
X_train = np.array(data_train['value'].tolist())  # 'value' 열에서 배열로 변환
y_train = data_train['target'].values  # 'target' 열

# 테스트데이터 화이트 가우시안 노이즈 첨가 부분
noise = 0.5
normal_test_data = normal_test_data + np.random.normal(0, noise, normal_test_data.shape)
bearing_fault_test_data = bearing_fault_test_data + np.random.normal(0, noise, bearing_fault_test_data.shape)
rotor_fault_test_data = rotor_fault_test_data + np.random.normal(0, noise, rotor_fault_test_data.shape)

# 테스트 데이터 처리
FFT_Transform_start_time = time.time()
normal_test_peak_amplitudes = process_segments(normal_test_data)
FFT_Transform_end_time = time.time()
bearing_fault_test_peak_amplitudes = process_segments(bearing_fault_test_data)
rotor_fault_test_peak_amplitudes = process_segments(rotor_fault_test_data)
FFT_Transform_time = FFT_Transform_end_time - FFT_Transform_start_time

normal_test_100 = [(i['fft_result'], 0) for i in normal_test_peak_amplitudes]
bearing_fault_test_100 = [(i['fft_result'], 1) for i in bearing_fault_test_peak_amplitudes]
rotor_fault_test_100 = [(i['fft_result'], 2) for i in rotor_fault_test_peak_amplitudes]

data_test_100 = normal_test_100 + bearing_fault_test_100 + rotor_fault_test_100
np.random.shuffle(data_test_100)  # 테스트 데이터 섞기

data_test = pd.DataFrame(data_test_100, columns=['fft_result', 'target'])
data_test['value'] = data_test['fft_result']

# X_test 및 y_test 정의
X_test = np.array(data_test['value'].tolist())  # 'value' 열에서 배열로 변환
y_test = data_test['target'].values  # 'target' 열

# 데이터 정규화 (MinMaxScaler 사용)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)  # 훈련 데이터 정규화
X_test = scaler.transform(X_test)  # 테스트 데이터 정규화

# 입력 형태 변경 (Conv1D 레이어에 맞추기 위해)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# One-Hot 인코딩
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()

# CNN 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 함수 정의 (배치별 loss 기록)
class LossHistory(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        batch_loss_history.append(logs['loss'])

# 모델 훈련
batch_loss_history = []
history = model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1, callbacks=[LossHistory()])

# 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# 혼동 행렬 및 시각화
Test_start_time = time.time()  # 시작 시간 기록
y_test_pred_classes = np.argmax(model.predict(X_test), axis=1)
Test_end_time = time.time()  # 종료 시간 기록
Test_time = Test_end_time - Test_start_time  # 소요 시간 계산
Total_time = FFT_Transform_time + Test_time
print(f"총 소요 시간 : FFT 변환 + 테스트 시간 = {FFT_Transform_time:.4f}s + {Test_time:.4f}s = {Total_time:.4f}s")

y_test_true_classes = np.argmax(y_test, axis=1)

confusion_mtx_test = confusion_matrix(y_test_true_classes, y_test_pred_classes)

fig, ax = plt.subplots(figsize=(5, 5))
disp_test = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx_test, display_labels=encoder.categories_[0])
disp_test.plot(ax=ax, cmap=plt.cm.Blues)
plt.show()

# 손실 그래프 시각화
plt.figure(figsize=(12, 5))
plt.plot(batch_loss_history, label='Batch Loss', color='green')
plt.title('Batch Loss Over All Epochs')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.legend()
plt.show()
