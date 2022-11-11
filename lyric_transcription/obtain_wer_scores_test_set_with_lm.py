from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from pyctcdecode import build_ctcdecoder
import kenlm
import argparse
import torch
from datasets import load_dataset, load_metric, Audio, Value, Features
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ProcessorWithLM


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


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


def map_to_result_lm(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits

    # pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor_with_lm.batch_decode(logits.cpu().detach().numpy()).text
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)

    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='obtain_wer_scores_test_set_with_lm.py',
        description='Obtain the WER scores on the valid and test sets'
    )

    parser.add_argument("--val_csv")  # path to csv with DSing and N20EM val information
    parser.add_argument("--dsing_test_csv")  # path to csv with DSing test set
    parser.add_argument("--nem_test_csv")  # path to csv with N20EM test set
    parser.add_argument("--model_checkpoint")  # path to csv with N20EM test set
    parser.add_argument("--lm_weights")  # path to csv with N20EM test set


    args = parser.parse_args()

    features = Features(
        {
            "text": Value("string"),
            "file": Value('string'),
            "audio": Audio(sampling_rate=16000)
        }
    )

    dataset_transcription = load_dataset(
        'csv', data_files={
            'valid': args.val_csv,
            'dsing_test': args.dsing_test_csv,
            'n20em_test': args.nem_test_csv
        },
    )

    dataset_transcription = dataset_transcription.cast(features)

    processor = Wav2Vec2Processor.from_pretrained(args.model_checkpoint)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_checkpoint,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    processor = Wav2Vec2Processor.from_pretrained(args.model_checkpoint)

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    dataset_transcription = dataset_transcription.map(prepare_dataset, num_proc=8)

    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    # build the decoder
    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=args.lm_weights,
    )

    # create a processor with the decoder
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )

    wer_metric = load_metric("wer")

    val_results = dataset_transcription["valid"].map(map_to_result_lm,
                                                     remove_columns=dataset_transcription["valid"].column_names,
                                                     load_from_cache_file=False)
    val_wer = wer_metric.compute(predictions=val_results["pred_str"], references=dataset_transcription["valid"]["text"])

    dsing_test_results = dataset_transcription["dsing_test"].map(map_to_result_lm, remove_columns=dataset_transcription[
        "dsing_test"].column_names, load_from_cache_file=False)
    dsing_test_wer = wer_metric.compute(predictions=dsing_test_results["pred_str"],
                                        references=dataset_transcription["dsing_test"]["text"])

    nem_test_results = dataset_transcription["n20em_test"].map(map_to_result_lm, remove_columns=dataset_transcription[
        "n20em_test"].column_names, load_from_cache_file=False)
    nem_test_wer = wer_metric.compute(predictions=nem_test_results["pred_str"],
                                      references=dataset_transcription["n20em_test"]["text"])

    scores = [[val_wer, dsing_test_wer, nem_test_wer]]
    df_scores = pd.DataFrame(scores)
    df_scores.columns = ["Validation Set WER", "DSing Test WER", "N20EM Test WER"]
    df_scores.to_csv("wav2vec2_only_scores_lm.csv")
