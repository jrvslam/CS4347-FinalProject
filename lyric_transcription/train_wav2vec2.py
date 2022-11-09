import json
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
import argparse

import torch
from datasets import load_dataset, load_metric, Audio, Value, Features
from transformers import Trainer, TrainingArguments, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, \
    Wav2Vec2Processor


# REFERENCE: Code for training the wav2vec2 model was referred/taken from the following links:
# https://huggingface.co/blog/fine-tune-wav2vec2-english
# https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def extract_all_chars(batch):
    """
    Method for extracting the characters from the vocabulary
    Args: batch
    Returns: dict
    """
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def prepare_dataset(batch):
    """
    Method for preparing the dataset by loading the audio files
    The dataloader lazy loads the audio file by reading from the file path

    Returns: batch
    """
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


def compute_metrics(pred):
    """
    Method for calculating the Word Error Rate
    Returns: dict
    """
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train_wav2vec2.py',
        description='Fine Tune a Wav2Vec2 model Lyric Transcription'
    )

    parser.add_argument("--train_csv")  # path to csv with DSing and N20EM train information
    parser.add_argument("--val_csv")  # path to csv with DSing and N20EM val information
    # parser.add_argument("--dsing_csv")  # path to csv with DSing test information
    # parser.add_argument("--nem_csv")  # path to csv with N20EM test information
    parser.add_argument("--checkpoint_folder")  # path to save the model checkpoints to

    args = parser.parse_args()

    # Define template of dataloader
    features = Features(
        {
            "text": Value("string"),
            "file": Value('string'),
            "audio": Audio(sampling_rate=16000)
        }
    )

    # load the files with information about transcription and corresponding audio files
    dataset_transcription = load_dataset(
        'csv', data_files={
            'train': args.train_csv,
            'valid': args.val_csv,
        },
    )

    dataset_transcription = dataset_transcription.cast(features)

    vocabs = dataset_transcription.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                       remove_columns=dataset_transcription.column_names["train"])

    chars_init = list(set(vocabs["train"]["vocab"][0]))
    chars = [x for x in chars_init if not x.isnumeric()]

    vocab_dict = {v: k for k, v in enumerate(chars) if not v.isnumeric()}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab_test.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer("./vocab_test.json", unk_token="[UNK]", pad_token="[PAD]",
                                     word_delimiter_token="|")

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=False)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    dataset_transcription = dataset_transcription.map(prepare_dataset,
                                                      remove_columns=dataset_transcription.column_names["train"],
                                                      num_proc=4)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = load_metric("wer")

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir="wv_test",
        group_by_length=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=10,
        fp16=True,
        gradient_checkpointing=False,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=10,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_transcription["train"],
        eval_dataset=dataset_transcription["valid"],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    processor.save_pretrained(save_directory=args.checkpoint_folder)
