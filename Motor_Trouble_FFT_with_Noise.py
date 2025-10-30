import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.fft import fft, fftfreq

# 데이터 로드
normal_data = pd.read_csv('정상상태_100개씩_100쌍.csv', encoding='utf-8')
bearing_fault_data = pd.read_csv('베어링고장_100개씩_100쌍.csv', encoding='utf-8')
rotor_fault_data = pd.read_csv('회전자고장_100개씩_100쌍.csv', encoding='utf-8')

# 각 데이터의 value 컬럼 추출
normal_value_data = normal_data['value'].values
bearing_fault_value_data = bearing_fault_data['value'].values
rotor_fault_value_data = rotor_fault_data['value'].values

def perform_fft(value_data):
    t = np.linspace(0, 100, 10000, endpoint=False)  # 시간 축 생성
    signal = np.zeros(10000)  # 빈 신호 배열
    
    # 주파수에 따라 사인파 생성
    for frequency in value_data:
        signal += np.sin(2 * np.pi * frequency * t) 
      
    signal_fft = np.abs(fft(signal))
    signal_fft[0] = 0  # DC 성분 제거
    
    frequency_bins = fftfreq(len(signal_fft), d=1/10000)  # 주파수 축 생성
    half_length = len(signal_fft) // 2  # 양의 주파수 부분만 사용

    # 진폭 기준으로 정렬하여 상위 5개 주파수 선택
    top_5_frequencies = sorted(range(half_length), key=lambda i: signal_fft[i], reverse=True)[:5]

    return signal_fft, [frequency_bins[i] for i in top_5_frequencies]

# 노이즈 추가
normal_value_data = normal_value_data + np.random.normal(0, 0.1, normal_value_data.shape)
bearing_fault_value_data = bearing_fault_value_data + np.random.normal(0, 0.1, bearing_fault_value_data.shape)
rotor_fault_value_data = rotor_fault_value_data + np.random.normal(0, 0.1, rotor_fault_value_data.shape)

# 100개의 FFT 결과 획득
bearing_fault_fft, frequency = perform_fft(bearing_fault_value_data)

print('[Top 5 frequencies]\n')
for i in range(1,6):
    print('Point ',i,' : ',frequency[i-1])

plt.plot(bearing_fault_fft)  # 첫 번째 세그먼트의 FFT 결과
plt.title('Bearing Fault FFT (Segment 1)')
plt.xlim(0,1000)


plt.tight_layout()
plt.show()
