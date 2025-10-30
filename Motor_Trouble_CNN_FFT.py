import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from scipy.signal import butter, lfilter  # 필터를 위한 라이브러리

# 데이터 로딩
normal_data = pd.read_csv('정상상태_100개씩_100쌍.csv', encoding='utf-8')
bearing_fault_data = pd.read_csv('베어링고장_100개씩_100쌍.csv', encoding='utf-8')
rotor_fault_data = pd.read_csv('회전자고장_100개씩_100쌍.csv', encoding='utf-8')

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff=5.0, fs=100.0, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# FFT 수행하는 함수 정의
def perform_fft(data_segment):
    data_segment = lowpass_filter(data_segment['value'])
    data_segment = pd.DataFrame(data_segment, columns=['value'])
    data_segment['value'] -= data_segment['value'].mean()  # DC 성분 제거
    sampling_rate = 100
    n = len(data_segment['value'])
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_values = np.fft.fft(data_segment['value'])
    amplitude = np.abs(fft_values)
    return pd.DataFrame({'Frequency': frequencies, 'Amplitude': amplitude})

# 피크 주파수와 진폭 추출 함수
def extract_peaks(df, num_peaks=2):
    peaks_indices = np.argsort(df['Amplitude'].values)[-num_peaks:]  # 진폭 기준 상위 n개 인덱스
    peak_frequencies = df['Frequency'].values[peaks_indices]
    peak_amplitudes = df['Amplitude'].values[peaks_indices]
    return peak_frequencies, peak_amplitudes

# 100개씩 데이터를 나누고 각 세그먼트에 대해 FFT 수행
def process_segments(data, segment_size=100):
    num_segments = len(data) // segment_size
    segments = []
    for i in range(num_segments):
        segment = data.iloc[i * segment_size:(i + 1) * segment_size]
        fft_result = perform_fft(segment)
        peak_frequencies, peak_amplitudes = extract_peaks(fft_result)
        segments.append({'fft_result': fft_result, 'peak_frequencies': peak_frequencies, 'peak_amplitudes': peak_amplitudes})
    return segments

# 훈련 및 테스트 데이터 분할
normal_train_data, normal_test_data = train_test_split(normal_data, test_size=0.3, random_state=42)
bearing_fault_train_data, bearing_fault_test_data = train_test_split(bearing_fault_data, test_size=0.3, random_state=42)
rotor_fault_train_data, rotor_fault_test_data = train_test_split(rotor_fault_data, test_size=0.3, random_state=42)

# 훈련 데이터 처리
normal_train_peak_amplitudes = process_segments(normal_train_data)
bearing_fault_train_peak_amplitudes = process_segments(bearing_fault_train_data)
rotor_fault_train_peak_amplitudes = process_segments(rotor_fault_train_data)

normal_train_100 = [(i['fft_result'], 0) for i in normal_train_peak_amplitudes]
bearing_fault_train_100 = [(i['fft_result'], 1) for i in bearing_fault_train_peak_amplitudes]
rotor_fault_train_100 = [(i['fft_result'], 2) for i in rotor_fault_train_peak_amplitudes]

data_train_100 = normal_train_100 + bearing_fault_train_100 + rotor_fault_train_100
data_train = pd.DataFrame(data_train_100, columns=['fft_result', 'target'])
'''
# 테스트데이터 화이트 가우시안 노이즈 첨가 부분
noise = 0.2
normal_test_data = normal_test_data + np.random.normal(0, noise, normal_test_data.shape)
bearing_fault_test_data = bearing_fault_test_data + np.random.normal(0, noise, bearing_fault_test_data.shape)
rotor_fault_test_data = rotor_fault_test_data + np.random.normal(0, noise, rotor_fault_test_data.shape)
'''
'''
# 테스트데이터 직류성분 노이즈 첨가 부분  Offset = 0.05, 0.1, 0.5, 1.0
bias = 1.0
normal_test_data = normal_test_data + bias
bearing_fault_test_data = bearing_fault_test_data + bias
rotor_fault_test_data = rotor_fault_test_data + bias
'''

# 테스트데이터 처리
normal_test_peak_amplitudes = process_segments(normal_test_data)
bearing_fault_test_peak_amplitudes = process_segments(bearing_fault_test_data)
rotor_fault_test_peak_amplitudes = process_segments(rotor_fault_test_data)

normal_test_100 = [(i['fft_result'], 0) for i in normal_test_peak_amplitudes]
bearing_fault_test_100 = [(i['fft_result'], 1) for i in bearing_fault_test_peak_amplitudes]
rotor_fault_test_100 = [(i['fft_result'], 2) for i in rotor_fault_test_peak_amplitudes]

data_test_100 = normal_test_100 + bearing_fault_test_100 + rotor_fault_test_100
data_test = pd.DataFrame(data_test_100, columns=['fft_result', 'target'])


import numpy as np

def pad_channel(data, target_shape):
    # 데이터가 리스트인 경우 NumPy 배열로 변환
    if isinstance(data, list):
        data = np.array(data)
        
    # target_shape에 맞게 패딩
    padded_data = np.zeros(target_shape)
    padded_data[:data.shape[0], :data.shape[1]] = data  # 기존 데이터로 패딩된 배열을 채움
    return padded_data

def create_input_data(data_segments):
    X = []
    for _, row in data_segments.iterrows():
        fft_df = row['fft_result']
        peak_frequencies, peak_amplitudes = extract_peaks(fft_df)

        # 주파수-진폭 데이터
        freq_amp_data = np.column_stack((fft_df['Frequency'].values, fft_df['Amplitude'].values)) 

        # 피크 주파수와 진폭 데이터
        peaks_data = [np.column_stack((peak_frequencies[i], peak_amplitudes[i])) for i in range(len(peak_frequencies))]
        
        # 각 피크 데이터를 패딩
        peaks_data_padded = [pad_channel(peaks, (100, 2)) for peaks in peaks_data]

        # 결합하여 4D 배열 생성 (주파수-진폭, 피크 주파수-피크 진폭)
        X.append([freq_amp_data] + peaks_data_padded)
    return np.array(X)

# 입력 데이터 생성
X_train = create_input_data(data_train)
X_train = np.transpose(X_train, (0, 2, 3, 1))
y_train = data_train['target'].values
X_test = create_input_data(data_test)
X_test = np.transpose(X_test, (0, 2, 3, 1)) 
y_test = data_test['target'].values

from sklearn.preprocessing import MinMaxScaler

# X_train과 X_test의 크기 확인
num_samples_train, num_channels, num_features, num_dimensions = X_train.shape
num_samples_test = X_test.shape[0]

# MinMaxScaler를 각각의 채널에 대해 적용하기 위해 reshape
X_train_reshaped = X_train.reshape(-1, num_features * num_dimensions)
X_test_reshaped = X_test.reshape(-1, num_features * num_dimensions)

# 정규화
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train_reshaped)
X_test_normalized = scaler.transform(X_test_reshaped)

# 정규화된 데이터를 원래 형태로 변환
X_train = X_train_normalized.reshape(num_samples_train, num_channels, num_features, num_dimensions)
X_test = X_test_normalized.reshape(num_samples_test, num_channels, num_features, num_dimensions)

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y_train = np.array(y_train).reshape(-1, 1)
y_train = encoder.fit_transform(y_train).toarray()
y_test = np.array(y_test).reshape(-1, 1)
y_test = encoder.transform(y_test).toarray()

def create_cnn_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # 주파수-진폭 채널
    x1 = tf.keras.layers.Conv2D(32, (3, 2), activation='relu', padding='same')(inputs)
    x1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(x1)
    x1 = tf.keras.layers.Conv2D(32, (3, 2), activation='relu', padding='same')(x1)
    x1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(x1)

    # 피크1-진폭 채널
    x2 = tf.keras.layers.Conv2D(32, (3, 2), activation='relu', padding='same')(inputs)
    x2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(x2)
    x2 = tf.keras.layers.Conv2D(32, (3, 2), activation='relu', padding='same')(x2)
    x2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(x2)

    # 피크2-진폭 채널
    x3 = tf.keras.layers.Conv2D(32, (3, 2), activation='relu', padding='same')(inputs)
    x3 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(x3)
    x3 = tf.keras.layers.Conv2D(32, (3, 2), activation='relu', padding='same')(x3)
    x3 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(x3)

    x1_flat = tf.keras.layers.Flatten()(x1)
    x2_flat = tf.keras.layers.Flatten()(x2)
    x3_flat = tf.keras.layers.Flatten()(x3)

    output1 = tf.keras.layers.Dense(3, activation='softmax')(x1_flat)  
    output2 = tf.keras.layers.Dense(3, activation='softmax')(x2_flat) 
    output3 = tf.keras.layers.Dense(3, activation='softmax')(x3_flat) 

    combined_outputs = tf.keras.layers.concatenate([output1, output2, output3])
    final_output = tf.keras.layers.Dense(3, activation='softmax')(combined_outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=final_output)
    return model


# 모델 생성 및 컴파일
input_shape = (100, 2, 3)  # 3개의 채널
model = create_cnn_model(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=50, batch_size=16)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# 결과 시각화
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 혼돈 행렬 생성
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',annot_kws={"size": 16})
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
