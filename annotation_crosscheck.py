import random
from pathlib import Path
import shutil

#randomly samples 180 clips for re-annotations to test cross-annotator accuracy

folder = Path("/Users/jamesshortland/Desktop/masters_final_annotations/re-annotation_070516/recordings")

all_files = list(folder.glob("*"))


files_to_keep = set(random.sample(all_files, 180))

kept_folder = folder / "kept_180"
kept_folder.mkdir(exist_ok=True)

for f in files_to_keep:
    shutil.move(str(f), kept_folder / f.name)

print(f"Moved {len(files_to_keep)} files to {kept_folder}. The rest remain in {folder}.")
