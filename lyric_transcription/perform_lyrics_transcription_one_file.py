import argparse

from datasets import Audio
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from pyctcdecode import build_ctcdecoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='perform_lyrics_transcription_one_file.py',
        description='Perform Lyrics Transcription on one file'
    )

    parser.add_argument("--checkpoint_path")  # path to model checkpoint
    parser.add_argument("--lm_model_file")  # path to n-gram .arpa file generated with KenLM
    parser.add_argument("--audio_path")  # path to audio file

    args = parser.parse_args()

    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint_path)
    lyrics_model = Wav2Vec2ForCTC.from_pretrained(
        args.checkpoint_path,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

    # build the decoder
    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=args.lm_model_file,
    )

    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )

    audio_dataset = Dataset.from_dict({"audio": [args.audio_path]}).cast_column("audio", Audio())
    input_features = processor(audio_dataset[0]["audio"]["array"], return_tensors="pt").input_values
    logits = lyrics_model(input_features).logits
    transcription = processor_with_lm.batch_decode(logits.detach().numpy()).text

    print("The Transcription is: " + transcription[0])

    with open("transcribed_output.txt", "w") as f:
        f.write(transcription[0])
