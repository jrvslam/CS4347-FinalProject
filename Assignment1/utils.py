import os
import librosa
import soundfile as sf

import warnings
warnings.filterwarnings('ignore')

data_dir = './data/test/100'
wav_path = os.path.join(data_dir, "Mixture.mp3")
            
y, sr = librosa.core.load(wav_path, sr=None, mono=True)
if sr != 44100:
    y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)

sf.write('./100.wav', y, 44100)