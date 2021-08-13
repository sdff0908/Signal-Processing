```python
from scipy.io import wavfile
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')
```

### 1. 주파수 차단

```python
def cutoff_freq_filter(data, sampling_rate, cut_frequency):
  freq = np.fft.rfftfreq(len(data), d=1/sampling_rate)
  fft_sig = np.fft.rfft(data)
  fft_sig[freq > cut_frequency] = 0
  cutoff_sig = np.fft.irfft(fft_sig)
  return cutoff_sig 

# data : np.array
# sampling rate : 1초당 가져오는 samples
```

(1) 푸리에 변환(rfft) 

(2) 일정 주파수 이상의 값을 0으로 설정 

(3) 푸리에 역변환(irfft)하여 time-domain 신호로 바꿔준다

### 2. waveplot그리기

```python
# parameter setting
sampling_rate = 10000

# load speech signal
sig, sr = librosa.load('/content/drive/MyDrive/sam.m4a', sr=sampling_rate)

# x_axis: time
t = np.arange(0, len(sig)/sr, 1/sr) 
plt.figure(1, figsize=(12,2))
plt.title('/sam/')
plt.xlabel('time(s)')
plt.ylabel('amplitude')
plt.plot(t, sig)
plt.show()

# x_axis: sample number
n = np.arange(0, len(sig), 1)
plt.figure(2, figsize=(12,2))
plt.title('/sam/')
plt.xlabel('sample #')
plt.ylabel('amplitude')
plt.plot(n, sig)
plt.show()
```
![다운로드 (13)](https://user-images.githubusercontent.com/72610879/129423016-20eceee4-2b4e-4153-8eeb-3df4b99ad282.png)

![다운로드 (14)](https://user-images.githubusercontent.com/72610879/129423028-985f9d09-ceb8-411d-8635-53317e74e003.png)


### 3. window

speech signal은 여러 phoneme이 시간에 따라 변화

이를 단구간으로 쪼개어 살펴보기 위해 window 사용

```python
# plot hamming window 
length = 256
h_window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (length - 1))

plt.plot(n, h_window)
plt.xlim(0, 256)
plt.xlabel('samples')
plt.ylabel('magnitude')
plt.title('Hamming window')
plt.show()

# hamming window with fourier transform 
n_fft = 4096 
fft_sig = np.fft.rfft(h_window, n=n_fft)
ang_freq = np.linspace(0, np.pi, len(fft_sig))
db = 20 * np.log10(fft_sig/fft_sig.max())

plt.plot(ang_freq, db)
plt.hlines(-3, xmin=0, xmax=ang_freq.max(), color='red', linestyles='--')
plt.xlabel('rad/samples')
plt.ylabel('dB magnitude')
plt.xlim(0,0.1)
plt.grid()
plt.show()
```


![다운로드 (8)](https://user-images.githubusercontent.com/72610879/129422169-e583d5b6-759c-4c25-9f7b-6acc68d07cdd.png)
![다운로드 (10)](https://user-images.githubusercontent.com/72610879/129422515-897e962c-2abe-4c82-bd46-9d6494e55149.png)


(1) n_fft : 몇 개의 샘플을 푸리에 변환할지 결정한다. 이때 n_fft > (푸리에 변환할 신호의 길이) 이어야 한다. 컴퓨터 연산 속도를 위해 주로 2의 제곱수를 사용한다.

n_fft >= (푸리에 변환할 신호의 길이) : 신호의 모든 샘플에 대해서 푸리에 변환이 이루어져야 하므로 일대일대응이 기본적으로 이루어져야 한다. 신호의 길이를 넘어선 값에 대해서는 zero padding이 이루어진다.

n_fft <  (푸리에 변환할 신호의 길이) : 전체 신호에 대해 푸리에 변환이 이루어지지 않는다. 신호 왜곡 발생

(2) dB

<img width="237" alt="3" src="https://user-images.githubusercontent.com/72610879/129423760-f19620ba-84ee-46f4-b769-78cdcf4a86a2.PNG">

signal/signal.max() : normalization, dB단위에서 신호의 최댓값을 0으로 만들어준다.

-3dB : 처음 신호의 에너지가 절반이 되는 지점으로, 3dB bandwidth는 이때의 주파수를 가리킨다. 위 그래프에서 3dB bandwidth는 약 0.04 rad/samples 

(3) np.fft.rfft()

np.fft.fft()는 (-pi, pi) 범위에 대해서 푸리에 변환. 하지만 현실에서 음의 주파수는 필요하지 않고, 실수범위에 대해서 푸리에 변환했을 때 그래프는 y축 대칭이다. 따라서 (0, pi) 범위에 대해서만 푸리에 변환하는 np.fft.rfft()를 사용한다.

(4) np.fft.fftfreq()

frequency 범위로 주파수 나타낸다. 위 코드에서는 fftfreq()를 사용하지 않고 (0, pi) 범위에 대한 angular frequency를 나타내었다. 

<img width="294" alt="2" src="https://user-images.githubusercontent.com/72610879/129423782-687cd423-7a1b-4282-97cd-27577cd8463d.PNG">

### 4. Short-Time Energy(STE)

<img width="224" alt="1" src="https://user-images.githubusercontent.com/72610879/129423807-2409da60-9d3a-48cb-b0e2-d79644debef6.PNG">

s(m) : speech signal

w(m) : window function

```python
def STE(signal):
  energy_mag = np.array([])
  i = 0
  while i < len(signal):
    if (i+frame_length) > len(signal):
      temp = np.concatenate((signal[i:], np.zeros(i+frame_length-len(signal))))
      windowed_frame = temp * h_window
    else:
      windowed_frame = signal[i:(i+frame_length)] * h_window
    power_sig = sum((windowed_frame)**2)
    energy_mag = np.append(energy_mag, power_sig)
    i += hop_length
  return energy_mag
```

```python
# load signal
sig, sr = librosa.load('/content/drive/MyDrive/sa.m4a', sr=10000)

# set parameters
frame_length = 256
overlap_rate = 0.5
hop_length = int(frame_length * (1-overlap_rate)) # 128

# window function
n = np.linspace(0,255,256)
h_window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (frame_length - 1))

# plot STE 
energy_mag = STE(sig)
t = np.linspace(0, len(sig)/sr, len(energy_mag)) 

plt.figure(figsize=(14,2))
plt.plot(t, energy_mag)
plt.title('Short-Time Energy')
plt.ylabel('energy magnitude')
plt.xlabel('time(s)')
plt.xlim(0.8,1.3)
plt.grid()
plt.show()
```

len(energy_mag)) = len(sig) / hop_length , 이 값이 소수점으로 떨어지면 +1


![다운로드](https://user-images.githubusercontent.com/72610879/129422198-ca6be0c3-6b25-4fd0-8bc3-4ff39e7d9bb2.png)

유성음 구간에서 STE 값 > 무성음 구간에서 STE 값(0에 근사) → 유성음과 무성음 구분 가능

### 5. Zero-Crossing Rate(ZCR)

```python
def sign_fn(array):
  sign_array = np.zeros(len(array))
  for i in range(len(array)):
    if array[i] >= 0:
      sign_array[i] = 1
    else:
      sign_array[i] = -1
  return sign_array
```

```python
def zero_cross(signal):
  x1 = signal
  x2 = np.append(np.zeros(1), signal[:-1])
  i = 0
  zcr = np.array([])
  while i < len(x1):
    try: 
      sub_x1 = x1[i:i+frame_length] * h_window
      sub_x2 = x2[i:i+frame_length] * h_window
      diff = sign_fn(sub_x1)-sign_fn(sub_x2)
      ab_diff = np.abs(diff)
      z = sum(ab_diff)/2
      zcr = np.append(zcr, z)
      i += hop_length
    except:
      break
  return zcr
```

try except 사용 이유: 루프를 돌다 보면 broadcasting 문제로 인해 sub_x1과 sub_x2 계산 불가하여 에러 발생. 이 때 샘플의 위치는 중요한 신호가 끝나고 노이즈만 발생하는 구간이므로 더이상 계산할 필요가 없음

```python
# load signal
sig, sr = librosa.load('/content/drive/MyDrive/sam.m4a', sr=10000)

# set parameters
frame_length = 256
overlap_rate = 0.5
hop_length = int(frame_length * (1-overlap_rate)) # 128

# calculate ZCR
zcr = zero_cross(sig)

# plot zcr
t = np.linspace(0, len(sig)/10000, len(zcr))
# t1 = np.linspace(0, len(sig)/10000, len(sig))

plt.figure(figsize=(14,2))
plt.plot(t, zcr)
plt.title('Zero-crossing Rate')
plt.xlabel('time(s)')
plt.ylabel('zcr')
plt.xlim(0.91, 1.2)
plt.grid()
plt.show()
```

<img width="292" alt="4" src="https://user-images.githubusercontent.com/72610879/129423818-cf592932-5f13-4f9d-860f-d15954dec529.PNG">

s(m) : speech signal

w(m) : window function

N : frame_length

n: frame 개수, len(sig) / hop_length, 이 값이 소수점으로 떨어지면 +1

Z(n) : n번째 window frame에 대한 식, 전체 ZCR 값은 {Z(0), Z(1), Z(2), ..., Z(n)}


![다운로드 (7)](https://user-images.githubusercontent.com/72610879/129422244-e611646d-f46d-4c09-a03c-f9ea18c7ff9b.png)

유성음 구간에서 ZCR값 < 무성음 구간에서 ZCR 값 → 유성음과 무성음 구분 가능

