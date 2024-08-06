

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import yfinance as yf

# Yahoo Finance'dan veri çekme
ticker = 'EURUSD=X'
data = yf.download(ticker, period='1y')
if data.empty:
    raise ValueError("Veri indirilemedi veya boş veri geldi. Lütfen veri kaynağını kontrol edin.")

df = data[['Close']]

# EMA 50 ve EMA 200 hesaplama
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# EMA 50 ve EMA 200 ortalamasını hesaplama
df['EMA_50_200_Avg'] = (df['EMA_50'] + df['EMA_200']) / 2

# Kalman Filtresi Parametreleri
transition_matrix = [[1]]
observation_matrix = [[1]]
initial_state_mean = df['Close'].iloc[0]
initial_state_covariance = 1.0
observation_covariance = 1.0
transition_covariance = 0.01

# Kalman Filtresinin Tanımlanması
kf = KalmanFilter(
    transition_matrices=transition_matrix,
    observation_matrices=observation_matrix,
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance
)

# Kapanış değerinin tahmini
state_means, state_covariances = kf.filter(df['Close'].values)
today_close_estimate = state_means[-1]

# EMA 50 ve EMA 200 ortalamasının tahmini
state_means_ema_avg, _ = kf.filter(df['EMA_50_200_Avg'].dropna().values)
today_ema_avg_estimate = state_means_ema_avg[-1]

# Yarının EMA 50 ve EMA 200 tahmini için yeni değerler ekleme
next_day_close = today_close_estimate  # Tahmini kapanış değeri
new_data = np.append(df['Close'].values, next_day_close)

# Yeni EMA 50 ve EMA 200 hesaplama
new_df = pd.DataFrame(new_data, columns=['Close'])
new_df['EMA_50'] = new_df['Close'].ewm(span=50, adjust=False).mean()
new_df['EMA_200'] = new_df['Close'].ewm(span=200, adjust=False).mean()

# Yarının EMA 50 ve EMA 200 tahmini
next_day_ema_50 = new_df['EMA_50'].iloc[-1]
next_day_ema_200 = new_df['EMA_200'].iloc[-1]

# Sonuçların Görselleştirilmesi
plt.figure(figsize=(15, 5))
plt.plot(df.index, df['Close'], label='Gerçek Kapanış Değerleri')
plt.plot(df.index, df['EMA_50'], label='EMA 50', color='blue')
plt.plot(df.index, df['EMA_200'], label='EMA 200', color='purple')
plt.plot(df.index[-len(state_means_ema_avg):], state_means_ema_avg, label='Kalman Filtresi EMA 50 ve 200 Ortalaması', color='red')
plt.axvline(df.index[-1], color='green', linestyle='--', label='Tahmin Edilen Gün')
plt.scatter(df.index[-1], today_ema_avg_estimate, color='green', label='Tahmin Edilen EMA 50 ve EMA 200 Ortalaması')
plt.legend()
plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.title('EUR/USD Kapanış Değerleri, EMA 50, EMA 200 ve Kalman Filtresi Tahminleri')
plt.show()

print(f"Tahmin Edilen Kapanış Değeri: {today_close_estimate}")
print(f"Tahmin Edilen EMA 50 ve EMA 200 Ortalaması: {today_ema_avg_estimate}")
print(f"Yarının EMA 50 Tahmini: {next_day_ema_50}")
print(f"Yarının EMA 200 Tahmini: {next_day_ema_200}")
