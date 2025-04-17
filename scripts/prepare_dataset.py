import os
import pandas as pd
from datasets import Dataset, DatasetDict

def load_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    data = []

    for _,row in df.iterrows():
        audio_path = row['audio_path']
        text = row['transcription']

        if os.path.exists(audio_path):
            data.append({'audio': audio_path, 'text':text})
        else:
            print(f"Warning: Audio file {audio_path} not found.")
    
    return data

def prepare_dataset():
    train_csv_path = 'data/GV_Train_100h/metadata.csv'
    dev_csv_path = 'data/GV_Dev_5h/metadata.csv'

    train_data = load_data_from_csv(train_csv_path)
    dev_data = load_data_from_csv(dev_csv_path)

    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': dev_dataset
    })

    dataset_dict.save_to_disk('data/processed_dataset')
    print("Dataset preparation complete. Saved to 'data/processed_dataset.' .")

if __name__ == '__main__':
    prepare_dataset()