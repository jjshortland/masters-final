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
