import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

# 데이터 로드
normal_data = pd.read_csv('정상상태_100개씩_100쌍.csv', encoding='utf-8')['value'].to_numpy()
bearing_fault_data = pd.read_csv('베어링고장_100개씩_100쌍.csv', encoding='utf-8')['value'].to_numpy()
rotor_fault_data = pd.read_csv('회전자고장_100개씩_100쌍.csv', encoding='utf-8')['value'].to_numpy()

# 노이즈 추가
noise = 1
normal_data = normal_data + np.random.normal(0, noise, normal_data.shape)
bearing_fault_data = bearing_fault_data + np.random.normal(0, noise, bearing_fault_data.shape)
rotor_fault_data = rotor_fault_data + np.random.normal(0, noise, rotor_fault_data.shape)

# 샘플링 설정
sampling_period = 0.01
time = np.arange(len(normal_data)) * sampling_period

# Haar 웨이블릿 변환 함수
def haar_transform(data):
    coeffs = pywt.wavedec(data, 'haar', level=5)  # Haar 웨이블릿 사용, 5단계 분해
    magnitude = [np.sum(np.abs(c)) for c in coeffs]  # 각 계수의 절댓값 합산
    return magnitude

# 데이터 변환
normal_magnitude = haar_transform(normal_data)
bearing_fault_magnitude = haar_transform(bearing_fault_data)
rotor_fault_magnitude = haar_transform(rotor_fault_data)

# 시각화
plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.bar(range(len(normal_magnitude)), normal_magnitude, label="Normal Data")
plt.title("Haar Wavelet Transform - Normal State")
plt.xlabel("Wavelet Coefficients Level")
plt.ylabel("Magnitude")
plt.legend()

plt.subplot(3, 1, 2)
plt.bar(range(len(bearing_fault_magnitude)), bearing_fault_magnitude, label="Bearing Fault Data", color='orange')
plt.title("Haar Wavelet Transform - Bearing Fault")
plt.xlabel("Wavelet Coefficients Level")
plt.ylabel("Magnitude")
plt.legend()

plt.subplot(3, 1, 3)
plt.bar(range(len(rotor_fault_magnitude)), rotor_fault_magnitude, label="Rotor Fault Data", color='green')
plt.title("Haar Wavelet Transform - Rotor Fault")
plt.xlabel("Wavelet Coefficients Level")
plt.ylabel("Magnitude")
plt.legend()

plt.tight_layout()
plt.show()
