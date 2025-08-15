import pandas as pd
import webrtcvad
from pydub import AudioSegment
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

model = load_silero_vad()
vad = webrtcvad.Vad()
vad.set_mode(3)


def silero_total_ms(timestamp, sr=16000):
    if not timestamp:
        return 0.0
    return sum(seg['end'] - seg['start'] for seg in timestamp) * 1000/sr


def silero_speech(path, sr=16000, threshold=0.30, neg_threshold=0.20, min_speech_duration_ms=60, speech_pad_ms=40):
    wav = read_audio(str(path), sampling_rate=sr)
    ts = get_speech_timestamps(wav, model, sampling_rate=sr, threshold=threshold, neg_threshold=neg_threshold,
                               min_speech_duration_ms=min_speech_duration_ms, speech_pad_ms=speech_pad_ms)
    if not ts:
        return 0
    return sum(seg['end'] - seg['start'] for seg in ts) * 1000 / sr


def webrtc_speech(path, frame_ms=30):
    try:
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        raw_audio = audio.raw_data
        frame_bytes = int(16000 * frame_ms / 1000) * 2
        # frames = [raw_audio[i:i + frame_bytes] for i in range(0, len(raw_audio), frame_bytes)]
        #
        # speech_frames = [vad.is_speech(frame, 16000) for frame in frames if len(frame) == frame_bytes]

        speech_ms = 0
        for i in range(0, len(raw_audio) - frame_bytes + 1, frame_bytes):
            frame = raw_audio[i:i+frame_bytes]
            if vad.is_speech(frame, 16000):
                speech_ms += frame_ms
        return speech_ms

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return 0


def classify_clip(path, frame_ms=30, webrtc_min_ms=250, silero_min_ms=250):
    w_ms = webrtc_speech(path, frame_ms=frame_ms)
    s_ms = silero_speech(path)
    if w_ms < webrtc_min_ms:
        return 'non_speech'
    return 'speech' if s_ms >= silero_min_ms else "non_speech"


def evaluate_ensemble(metadata_csv, audio_dir):
    metadata = pd.read_csv(metadata_csv)
    metadata = metadata[metadata['label'].isin(['speech', 'non_speech'])]
    paths = [Path(audio_dir) / fn for fn in metadata['filename']]
    y_true = metadata['label'].tolist()
    y_pred = [classify_clip(p) for p in paths]
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary',
                                                       pos_label='speech', zero_division=0)
    print("WebRTC ➜ Silero (filter) Evaluation:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=['non_speech', 'speech'], zero_division=0))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_true, y_pred, labels=['non_speech', 'speech']))


evaluate_ensemble('/Users/jamesshortland/Desktop/labels/complete_annotations.csv',
                  '/Users/jamesshortland/PycharmProjects/Masters_Final/five_second_paula_recordings')
