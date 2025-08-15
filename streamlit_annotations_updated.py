import streamlit as st
import pandas as pd
from pathlib import Path

# Constants
label_file = 'annotations.csv'
audio_dir = Path('recordings')
label_options = ['speech', 'non_speech']

# Load audio files
audio_files = sorted(audio_dir.glob("*.wav"))

# Load or create annotations
label_path = Path(label_file)
if label_path.exists():
    annotations = pd.read_csv(label_path)
else:
    annotations = pd.DataFrame(columns=['filename', 'label'])
    annotations.to_csv(label_file, index=False)

# Track state
if 'index' not in st.session_state:
    st.session_state.index = 0

# Filter files that aren't already labeled
labeled = set(annotations['filename'])
unlabeled_files = [f for f in audio_files if f.name not in labeled]

# Adjust index in case labels already exist
if st.session_state.index >= len(unlabeled_files):
    st.success("All files labeled!")
    st.stop()

# Display current file
current_file = unlabeled_files[st.session_state.index]

st.title("Audio Annotation Tool – One at a Time")
st.info("Thanks for helping me annotate my dataset! Please listen to each 5-second clip fully, then click either the"
        " speech or non-speech buttons. Once you've clicked a button, it will load the next audio file and save your"
        " answer. If you need to go back, you can press the go back button to change your answer. All answers are saved"
        " right away, so feel free to close this whenever and pick it back up later. If you're unsure of a clip, press "
        " the 'Flag' button so I know to check it myself. Whenever you're sick of doing this, send me the csv file "
        " that's saved in the same directory. Thanks again!")
st.info(f"Labeling file {st.session_state.index + 1} of {len(unlabeled_files)}")

st.markdown(f"**File: {current_file.name}**")
st.audio(current_file.read_bytes(), format="audio/wav")

# Layout buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("Speech"):
        # Save label
        annotations = pd.read_csv(label_file)
        annotations = pd.concat(
            [annotations, pd.DataFrame([[current_file.name, "speech"]], columns=['filename', 'label'])],
            ignore_index=True
        )
        annotations.to_csv(label_file, index=False)
        st.session_state.index += 1
        st.rerun()

with col2:
    if st.button("Non-Speech"):
        # Save label
        annotations = pd.read_csv(label_file)
        annotations = pd.concat(
            [annotations, pd.DataFrame([[current_file.name, "non_speech"]], columns=['filename', 'label'])],
            ignore_index=True
        )
        annotations.to_csv(label_file, index=False)
        st.session_state.index += 1
        st.rerun()

with col3:
    if st.button("Flag"):
        # Save label
        annotations = pd.read_csv(label_file)
        annotations = pd.concat(
            [annotations, pd.DataFrame([[current_file.name, "flagged"]], columns=['filename', 'label'])],
            ignore_index=True
        )
        annotations.to_csv(label_file, index=False)
        st.session_state.index += 1
        st.rerun()

