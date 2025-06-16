import soundfile as sf
data, samplerate = sf.read('freesound_dataset/speech/394632_Proverbs_13_verse_3.wav')
print(data.shape, samplerate)