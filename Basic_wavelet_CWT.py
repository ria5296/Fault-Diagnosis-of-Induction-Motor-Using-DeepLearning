import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt


def process_segments(data, segment_size=100):
    """데이터를 segment_size 크기로 분할"""
    num_segments = len(data) // segment_size
    segments = []
    for i in range(num_segments):
        segment = data[i * segment_size:(i + 1) * segment_size]
        segments.append(segment)
    return segments


def wavelet_transform(data, sampling_period=0.01, scales=np.arange(1, 32), wavelet='cmor'):
    """웨이블릿 변환 적용"""
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period=sampling_period)
    magnitude = np.sum(np.abs(coefficients), axis=0) 
    return magnitude


# 데이터 로드 및 노이즈 추가
normal_data = pd.read_csv('정상상태_100개씩_100쌍.csv', encoding='utf-8')['value'].to_numpy()
bearing_fault_data = pd.read_csv('베어링고장_100개씩_100쌍.csv', encoding='utf-8')['value'].to_numpy()
rotor_fault_data = pd.read_csv('회전자고장_100개씩_100쌍.csv', encoding='utf-8')['value'].to_numpy()

noise = 1
normal_data = normal_data + np.random.normal(0, noise, normal_data.shape)
bearing_fault_data = bearing_fault_data + np.random.normal(0, noise, bearing_fault_data.shape)
rotor_fault_data = rotor_fault_data + np.random.normal(0, noise, rotor_fault_data.shape)

# 데이터 분할
segment_size = 100
normal_segments = process_segments(normal_data, segment_size)
bearing_fault_segments = process_segments(bearing_fault_data, segment_size)
rotor_fault_segments = process_segments(rotor_fault_data, segment_size)

# 공통 설정
sampling_period = 0.01
scales = np.arange(1, 32)
wavelet = 'cmor'

# 시각화 함수
def plot_segments(segments, title_prefix, color, sampling_period=0.01, scales=np.arange(1, 32), wavelet='cmor'):
    plt.figure(figsize=(15, 12))
    for i, segment in enumerate(segments[:5]):  # 앞의 5개 segment만 시각화
        magnitude = wavelet_transform(segment, sampling_period, scales, wavelet)
        time = np.arange(len(segment)) * sampling_period

        plt.subplot(5, 2, 2 * i + 1)
        plt.plot(time, segment, marker='o', label=f"Segment {i+1}")
        plt.title(f"{title_prefix} - Segment {i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()

        plt.subplot(5, 2, 2 * i + 2)
        plt.plot(time, magnitude, color=color, label=f"Segment {i+1} - Wavelet Transform")
        plt.title(f"Wavelet Transform - {title_prefix} (Segment {i+1})")
        plt.xlabel("Time (s)")
        plt.ylabel("Magnitude")
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.show()


# 정상 상태 시각화
plot_segments(normal_segments, "Normal State", "orange", sampling_period, scales, wavelet)

# 베어링 고장 시각화
plot_segments(bearing_fault_segments, "Bearing Fault", "blue", sampling_period, scales, wavelet)

# 회전자 고장 시각화
plot_segments(rotor_fault_segments, "Rotor Fault", "green", sampling_period, scales, wavelet)
