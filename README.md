# Masters Final
The final project for my MSc Data Science and Artificial Intelligence degree. In collaboration with the **Sensing the Forest** organization, this project will aim to take a livestreamed audio soundscape of a forest and remove human voices, before publishing it to the website for the public to enjoy. 

## Dataset

The dataset was produced using the Freesound.org API. Audio clips were downloaded and converted into `.wav` files.

The following tags were used to produce a dataset split into two categories: **speech** and **non-speech**. In each case, up to 50 audio clips were attempted to be downloaded. Clips were limited to those between **5 seconds and 30 seconds** in duration, to control file size and ensure clips were long enough to be useful for model training.

The tags used were:

- **Speech**: `"human speech"`, `"talking"`, `"conversation"`, `"reading"`, `"narration"`, `"dialogue"`, `"interview"`, `"phone call"`, `"crowd talking"`, `"lecture"`, `"story telling"`, `"group conversation"`, `"presentation"`, `"speech"`, `"people talking"`, `"words"`, `"english"`
- **Non-Speech**: `"forest"`, `"rain"`, `"wind"`, `"river"`, `"birds"`, `"leaves"`, `"forest ambience"`, `"water stream"`, `"windy"`

Alongside this dataset, a CSV file was generated to track the **file name and location**, and to record which of the two categories each clip belongs to. A `verified` column allows manual verification that each file is correctly categorized based on its content.

The current dataset consists of **411 non-speech clips** and **436 speech clips**, for a total of **847 clips**. These were selected from an initial pool of **1173 clips** (751 speech and 422 non-speech) that were **manually verified**. A greater variety of speech clips were intentionally included, due to the wide range of **human speech characteristics and contexts** that the final model will need to handle, whereas the **non-speech domain** is comparatively more homogeneous. The **verification process** ensured that only clips genuinely containing human speech (or non-speech) were retained, excluding any that were **mis-tagged**.

Finally, the dataset was split into **train/validation/test** sets to enable a proper machine learning workflow. The split was performed at this stage to allow for consistent comparisons between different solutions. An **80% train**, **10% validation**, and **10% test** split was used. The split was stratified within the **speech** and **non-speech** categories to ensure that each subset maintained the same class balance.

The final dataset is as follows:

- **676 training clips:** 348 speech, 328 non-speech  
- **87 validation clips:** 45 speech, 42 non-speech  
- **84 test clips:** 43 speech, 41 non-speech



This dataset will serve as the input for training and comparing different **Voice Activity Detection (VAD)** models, with the goal of detecting and filtering human speech from environmental audio streams.

## Pre-Trained VAD Testing

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

| Model        | Accuracy | Precision | Recall | F1 Score |
|--------------|----------|-----------|--------|----------|
| **Silero**   | 0.921    | 0.970     | 0.874  | 0.919    |
| **Pyannote** | 0.515    | 0.515     | 1.000  | 0.680    |
| **Webrtc**   | 0.628    | 0.592     | 0.895  | 0.712    |


- **Silero** demonstrated strong performance, with a good balance of precision and recall. It sometimes missed subtle or low-volume speech, but generally made few mistakes.
- **Pyannote** showed perfect recall but very low precision — it classified *everything* as speech, leading to high false positives. This suggests an over-sensitive threshold or model tuning mismatch.
- **Webrtc** showed strong performance, but not as effective as Silero, only scoring higher in its recall score (89.5% vs 87.4%). This test was completed in VAD mode 3, the most forgiving setting. VAD mode 0-2 all produced the same results as pyannote, labeling all clips as speech.

## Bespoke Machine Learning VAD
The next step was to code a bespoke VAD model using traditional machine learning techniques. 

### MFCC SVM Model
A 2012 paper identified **Mel-Frequency Cepstral Coefficients (MFCCs)** combined with a **Support Vector Machine (SVM)** can be effective for voice activity detection. In this first attempt, **13 MFCCs** were extracted for each audio clip. An SVM was then trained on a combined **train + validation** dataset (following the same split as outlined in the `dataset` section). While this is a relatively simple model, it lays the groundwork for more advanced approaches. Despite this, this first model resulted in strong performance on the held-out test dataset:
- **Accuracy**: 0.798
- **Precision**: 0.760
- **Recall**: 0.884
- **F1 Score**: 0.817

This result is promising, especially for a lightweight model. The precision–recall balance indicates that the classifier correctly detects most speech clips while keeping false positives relatively low. While it doesn't reach perfect recall, the model captures the majority of speech instances and can serve as a solid foundation for further tuning or integration into an ensemble system.

### Iterative Improvements

#### 1. Feature Scaling  
SVMs are sensitive to feature scale, so normalization was introduced using a standard scaler. This alone yielded a marked improvement:

- **Accuracy**: 0.881  
- **Precision**: 0.884  
- **Recall**: 0.884  
- **F1 Score**: 0.884  

#### 2. Delta and Delta-Delta MFCC Features  
To enhance the representation of speech dynamics, **Delta** (first derivative) and **Delta-Delta** (second derivative) features were added. These capture how the MFCCs change over time, increasing the total features per clip from 13 to 39. Combined with normalization, this led to further gains:

- **Accuracy**: 0.905  
- **Precision**: 0.872  
- **Recall**: 0.954  
- **F1 Score**: 0.911  

While precision slightly decreased, the significant boost in recall is valuable for this task. In the context of VAD, **false negatives (missed speech)** are more harmful than **false positives**, so a higher recall is preferable.

### Model Comparison Table

| Model Version            | Features Used                   | Scaling | Accuracy  | Precision | Recall    | F1 Score  |
|--------------------------|----------------------------------|---------|-----------|-----------|-----------|-----------|
| **Base MFCC SVM**        | 13 MFCC                         | ❌      | 0.798     | 0.760     | 0.884     | 0.817     |
| **+ Scaler**             | 13 MFCC                         | ✅      | 0.881     | **0.884** | 0.884     | 0.884     |
| **+ Delta + Delta-Delta**| 13 MFCC + Δ + ΔΔ (total: 39)    | ✅      | **0.905** | 0.872     | **0.954** | **0.911** |




