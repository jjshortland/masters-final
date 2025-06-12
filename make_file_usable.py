from pydub import AudioSegment

file_path = '/Users/jamesshortland/Desktop/Master_Final_Project/raw_files/street_with_talking.wav'
audio = AudioSegment.from_file(file_path)
audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
audio.export('/Users/jamesshortland/Desktop/Master_Final_Project/workable_files/street_with_talking.wav', format="wav")

