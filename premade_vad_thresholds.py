import numpy as np
import pandas as pd
from pathlib import Path
from webRTC_silero_pipeline import webrtc_speech, silero_speech


df = pd.read_csv('/Users/jamesshortland/Desktop/labels/complete_annotations.csv')
df = df[df['label'] == 'speech'].copy()
paths = [Path('/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_paula_recordings') / f for f in df['filename']]

w_vals, s_vals = [], []
for p in paths:
    w_vals.append(webrtc_speech(p, frame_ms=20))  # your ms-returning helper
    s_vals.append(silero_speech(p, threshold=0.35, neg_threshold=0.25))

def q(arr): return {k: np.percentile(arr, k) for k in (5,10,25,50,75,90)}
print("WebRTC ms quantiles:", q(w_vals))
print("Silero ms quantiles:", q(s_vals))
print("Share of true-speech clips with Silero ms ≥ 250:", np.mean(np.array(s_vals) >= 250))
print("… ≥ 200:", np.mean(np.array(s_vals) >= 200))
print("… ≥ 150:", np.mean(np.array(s_vals) >= 150))