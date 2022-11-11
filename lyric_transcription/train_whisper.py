from dataclasses import dataclass
from typing import Any, Dict, List, Union
import argparse
import evaluate
import torch

from datasets import load_dataset, Audio, Value, Features
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import WhisperTokenizer


# Code for training whisper referred from https://huggingface.co/blog/fine-tune-whisper


def compute_metrics(pred):
    """
    Method for calculating the Word Error Rate
    Returns: dict
    """

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def prepare_dataset(batch):
    """
    Method to prepare dataset and resample dataset
    """

    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_whisper_tiny.py',
        description='Fine Tune a Whisper Tiny model for lyric transcription'
    )

    parser.add_argument("--train_csv")  # path to csv with DSing and N20EM train information
    parser.add_argument("--val_csv")  # path to csv with DSing and N20EM val information
    parser.add_argument("--checkpoint_folder")  # path to save the model checkpoints to

    args = parser.parse_args()

    features = Features(
        {
            "sentence": Value("string"),
            "file": Value('string'),
            "audio": Audio(sampling_rate=16000)
        }
    )

    sample_data = load_dataset(
        'csv', data_files={
            'train': args.train_csv,
            'valid': args.val_csv,
        },
    )

    sample_data = sample_data.cast(features)

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", language="english")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="English", task="transcribe")

    sample_data = sample_data.map(prepare_dataset, remove_columns=sample_data.column_names["train"], num_proc=8)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.checkpoint_folder,  # change to a repo name of your choice
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        num_train_epochs=10,
        warmup_steps=500,
        gradient_checkpointing=True,
        fp16=True,
        group_by_length=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        save_total_limit=10
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=sample_data["train"],
        eval_dataset=sample_data["valid"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
