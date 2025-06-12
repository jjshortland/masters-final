import freesound
import os
from pathlib import Path
from pydub import AudioSegment

API_key = 'LuLFjA93peZgs3gfxA2LRvm7f6HqubMfquhtfoRe'

client = freesound.FreesoundClient()
client.set_token(API_key, 'token')

def download_and_convert(query, label, save_dir, existing_filenames, max_results=5):
    results = client.text_search(query=query, fields='id,name,previews', page_size=max_results,
                                 filter="duration:[5 TO 30]")
    label_path = Path(save_dir) / label
    label_path.mkdir(parents=True, exist_ok=True)

    metadata_entries = []

    for sound in results:
        try:
            preview_url = sound.previews.preview_hq_mp3
            fname = f'{sound.id}_{sound.name.replace(' ', '_')}'
            mp3_path = label_path / f'{fname}.mp3'
            wav_path = label_path / f'{fname}.wav'

            if wav_path.name in existing_filenames:
                print(f"Skipping duplicate: {wav_path.name}")
                continue

            print(f'Downloading: {mp3_path.name}')
            os.system(f'curl -s {preview_url} -o \"{mp3_path}\"')

            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")
            mp3_path.unlink()

            metadata_entries.append([wav_path.name, label, "False"])

        except Exception as e:
            print(f"Failed to download/convert {sound.name}: {e}")

    return metadata_entries

def build_database():
    save_dir = 'freesound_dataset'
    existing_filenames = set()
    metadata_file = Path(save_dir) / 'metadata.csv'
    if metadata_file.exists():
        import csv
        with open(metadata_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_filenames.add(row["filepath"])
    save_dir = 'freesound_dataset'
    metadata_file = Path(save_dir) / 'metadata.csv'

    all_metadata = []
    tag_dict = {
        #  "speech": ["human speech", "talking", "conversation", "reading", "narration"],
        #  "speech": ["dialogue", "interview", "phone call", "crowd talking", "lecture"]
        #  "non_speech": ["forest", "rain", "wind", "river", "birds", "leaves"]
        #  "non_speech": ["windy"]
        #  "speech": ["narration", "reading", "story telling", "group conversation", "presentation"]
        "speech": ["speech", "people talking", "words", "english"]
    }

    for label, tag_list in tag_dict.items():
        for tag in tag_list:
            all_metadata += download_and_convert(tag, label, save_dir, existing_filenames, max_results=50)

    with open(metadata_file, "a", newline="") as f:
        writer = csv.writer(f)
        # Do not write header again when appending
        writer.writerows(all_metadata)

    print(f"Dataset built at: {save_dir}")
    print(f"Metadata saved to: {metadata_file}")


build_database()


