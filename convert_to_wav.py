import os
import subprocess

mp3_dir = 'data/GV_Train_100h/audio'
wav_dir = 'data/GV_Train_100h/wav'

os.makedirs(wav_dir, exist_ok=True)

for filename in os.listdir(mp3_dir):
    if filename.endswith('.mp3'):
        mp3_path = os.path.join(mp3_dir, filename)
        wav_filename = os.path.splitext(filename)[0] + '.wav'
        wav_path = os.path.join(wav_dir, wav_filename)
        subprocess.run([
            'ffmpeg', '-i', mp3_path,
            '-ar', '16000',
            '-ac', '1',
            wav_path
        ])
