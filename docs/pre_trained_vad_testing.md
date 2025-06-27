## Pre-Trained VAD Testing: Original Dataset

Three different pre-trained Voice Activity Detectors (VADs) were tested on a manually verified dataset containing both **speech** and **non-speech** audio clips. The goal was to evaluate their out-of-the-box performance in distinguishing human speech from environmental audio.

### Models Tested

- **Silero VAD**  
  A lightweight, efficient VAD model developed by Silero, designed for real-time applications and easy integration.
  
- **Pyannote VAD**  
  A more heavyweight model from the `pyannote-audio` library, trained with diarization and speaker segmentation tasks in mind. It uses a segmentation model to generate speech activity predictions over time.

- **Webrtc VAD**  
  WebRTC provides a lightweight, real-time Voice Activity Detection (VAD) system developed for use in web-based communications (like Google Meet), using rule-based logic to detect voiced audio in short frames of raw waveform data.

### Dataset & Evaluation

The models were tested using the verified audio clip dataset. Evaluation metrics included:
- **Accuracy** (overall correct predictions)
- **Precision** (how many predicted “speech” clips were actually speech)
- **Recall** (how many actual speech clips were correctly identified)
- **F1 Score** (harmonic mean of precision and recall)

### Results

| Model        | Accuracy  | Precision | Recall    | F1 Score  |
|--------------|-----------|-----------|-----------|-----------|
| **Silero**   | **0.921** | **0.970** | 0.874     | **0.919** |
| **Pyannote** | 0.515     | 0.515     | 1.000     | 0.680     |
| **Webrtc**   | 0.628     | 0.592     | **0.895** | 0.712     |


- **Silero** demonstrated strong performance, with a good balance of precision and recall. It sometimes missed subtle or low-volume speech, but generally made few mistakes.
- **Pyannote** showed perfect recall but very low precision — it classified *everything* as speech, leading to high false positives. This suggests an over-sensitive threshold or model tuning mismatch.
- **Webrtc** showed strong performance, but not as effective as Silero, only scoring higher in its recall score (89.5% vs 87.4%). This test was completed in VAD mode 3, the most forgiving setting. VAD mode 0-2 all produced the same results as pyannote, labeling all clips as speech.
