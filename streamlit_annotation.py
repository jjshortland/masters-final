import streamlit as st
import pandas as pd
from pathlib import Path

label_file = 'annotations.csv'
audio_dir = 'five_second_paula_recordings'
label_options = ['speech', 'non_speech']
batch_size = 20

label_path = Path(label_file)

audio_files = sorted(Path(audio_dir).glob("*.wav"))

st.title('Audio Annotation Tool')
st.info('Thank you for helping out! By default, you will go through 20 clips at a time. Each clip is 5 seconds long,'
        ' so each grouping of 20 clips will take you 1:40. There should be a lot more clips without speech than with,'
        ' so do not worry about that. After the clip plays, change the dropdown menu to speech or non speech. '
        ' Once you click Submit all Labels, the 20 clips will refresh. Please do as many as you can! Once you are done,'
        'send me the final csv file, which should be in the project directory, and be called "annotations.csv".'
        'Reach out to me if you have any questions!')

if label_path.exists():
    annotations = pd.read_csv(label_path)
    st.info('Loaded existing annotations file')
else:
    annotations = pd.DataFrame(columns=['filename', 'label'])
    st.info('No existing annotations file found, creating new file...')

labeled = set(annotations['filename'])
to_label = [f for f in audio_files if f.name not in labeled]

if not to_label:
    st.success("All files labeled!")
    st.stop()

batch = to_label[:batch_size]

st.title("Batch Audio Annotation")
st.write(f"Showing {len(batch)} of {len(to_label)} remaining files")


with st.form("annotation_form"):
    new_labels = []

    for i, audio_file in enumerate(batch):
        st.markdown(f"**{i+1}. File: {audio_file.name}**")
        st.audio(audio_file.read_bytes(), format='audio/wav')
        label = st.radio(
            f"Label for {audio_file.name}",
            label_options,
            key=f"radio_{audio_file.name}",
            horizontal=True
        )
        # label = st.selectbox(f"Label for {audio_file.name}", label_options, key=audio_file.name)
        new_labels.append((audio_file.name, label))
        st.markdown("---")

    submitted = st.form_submit_button("Submit All Labels")

    if submitted:
        new_df = pd.DataFrame(new_labels, columns=['filename', 'label'])
        annotations = pd.concat([annotations, new_df], ignore_index=True)
        annotations.to_csv(label_file, index=False)
        st.success(f"Saved {len(new_labels)} labels!")
        st.rerun()

