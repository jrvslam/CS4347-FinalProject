from speechbrain.pretrained import EncoderDecoderASR, EncoderASR
import torch


if __name__ == "__main__":
    
    asr_model = EncoderDecoderASR.from_hparams(
            source="trained_model",
            hparams_file = "hyperparams.yaml",
            savedir="pretrained_model"
            )
        
        
    transcribed_text = asr_model.transcribe_file("")
    
    print(transcribed_text)
