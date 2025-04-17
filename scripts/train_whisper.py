import torch
from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    #DataCollatorSpeechSeq2SeqWithPadding
)
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input features and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
# Load dataset
dataset = load_from_disk("data/processed_dataset")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Load processor and model
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name, language="Hindi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Preprocess function
def preprocess_function(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Apply preprocessing
processed_dataset = dataset.map(
    preprocess_function,
    remove_columns=dataset["train"].column_names,
    num_proc=4
)

# Data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# Evaluation metric
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="models/whisper-hindi-finetuned",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    save_strategy="epoch",
    learning_rate=1e-5,
    warmup_steps=500,
    logging_dir="logs",
    logging_steps=10,
    save_total_limit=2,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# Train
trainer.train()

# Save model and processor
trainer.save_model("models/whisper-hindi-finetuned")
processor.save_pretrained("models/whisper-hindi-finetuned")
