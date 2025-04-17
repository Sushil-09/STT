import os
import csv

text_file = 'data/GV_Train_100h/text'
wav_dir = 'data/GV_Train_100h/wav'
output_csv = 'data/GV_Train_100h/metadata.csv'

with open(text_file, 'r', encoding='utf-8') as tf, open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['audio_path', 'transcription'])

    for line in tf:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            utt_id, transcription = parts
            wav_path = os.path.join(wav_dir, f"{utt_id}.wav")
            if os.path.exists(wav_path):
                writer.writerow([wav_path, transcription])
