from dataclasses import dataclass
from typing import Any, Dict, List, Union
import argparse
import evaluate
import pandas as pd
import torch
from datasets import load_dataset, Audio, Value, Features
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration


# Code for finetuning the whisper model referred from https://huggingface.co/blog/fine-tune-whisper
# https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb


def prepare_dataset(batch):
    """
    Method to prepare dataset by resampling audio to 16kHz
    """
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


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


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_features"], device="cuda").unsqueeze(0)
        text = model.generate(input_values)

    batch["pred_str"] = processor.batch_decode(text, skip_special_tokens=True)
    batch["text"] = processor.decode(batch["labels"], skip_special_tokens=True)

    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='obtain_wer_scores_test_set_whisper.py',
        description='Obtain the WER scores on the valid and test sets'
    )

    parser.add_argument("--val_csv")  # path to csv with DSing and N20EM val information
    parser.add_argument("--dsing_test_csv")  # path to csv with DSing test set
    parser.add_argument("--nem_test_csv")  # path to csv with N20EM test set
    parser.add_argument("--model_checkpoint")  # path to csv with N20EM test set

    args = parser.parse_args()

    # load the template of the features to be processed
    features = Features(
        {
            "sentence": Value("string"),
            "file": Value('string'),
            "audio": Audio(sampling_rate=16000)
        }
    )

    # load the dataset which is present as csv format
    # The csv contains three columns,
    # sentence: the transcription of audio file
    # file: path to the audio file
    # audio: for now, it is path to audio file. When loaded, it is resampled
    # and loaded as raw audio

    dataset_transcription = load_dataset(
        'csv', data_files={
            'valid': args.val_csv,
            'dsing_test': args.dsing_test_csv,
            'n20em_test': args.nem_test_csv
        },
    )

    # Cast the data to the required template defined
    # The Audio class ensures resampling and lazy loading as required
    dataset_transcription = dataset_transcription.cast(features)

    # load the feature extractor, tokenizer and processor and the final model
    # which is finetuned on the dataset

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", language="english")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_checkpoint)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_transcription = dataset_transcription.map(prepare_dataset, num_proc=8)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")

    # Calculate the val set WER
    valid_results = dataset_transcription["valid"].map(map_to_result,
                                                       remove_columns=dataset_transcription["valid"].column_names,
                                                       load_from_cache_file=False)
    pred_text_valid = [x[0] for x in valid_results["pred_str"]]
    valid_test_wer = metric.compute(predictions=pred_text_valid, references=valid_results["text"])

    # Calculate the DSing test set WER
    dsing_test_results = dataset_transcription["dsing_test"].map(map_to_result,
                                                                 remove_columns=dataset_transcription[
                                                                     "dsing_test"].column_names,
                                                                 load_from_cache_file=False)
    pred_text_dsing = [x[0] for x in dsing_test_results["pred_str"]]
    dsing_test_wer = metric.compute(predictions=pred_text_dsing, references=dsing_test_results["text"])

    # Calculate the N20EM test set WER
    n20em_test_results = dataset_transcription["n20em_test"].map(map_to_result,
                                                                 remove_columns=dataset_transcription[
                                                                     "n20em_test"].column_names,
                                                                 load_from_cache_file=False)
    pred_text_n20em = [x[0] for x in n20em_test_results["pred_str"]]
    n20em_test_wer = metric.compute(predictions=pred_text_n20em, references=n20em_test_results["text"])

    # Save the results as a csv file
    results = [valid_test_wer, dsing_test_wer, n20em_test_wer]
    results_csv = pd.DataFrame(results)
    results_csv.columns = ["Val WER", "DSing Test WER", "N20EM Test WER"]
    results_csv.to_csv("Results_whisper.csv", index=False)
