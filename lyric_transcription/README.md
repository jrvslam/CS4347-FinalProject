# Lyric Transcription

This folder provides the code resources for training the Lyric Transcription Model, obtain the WER scores on the test sets and perform inference on a single file.


## Prepare the files for training the model:


    The Wav2Vec2 Training scripts require a csv with the following columns:

    a) text: The transcribed lyrics of a given audio file

    b) file: The path to the audio file

    c) audio: The path to the audio file again (internally, this is used by the module to load the audio file)


## Training the Wav2Vec2 model:
 
To train the Wav2Vec2 model, run:

```python train_wav2vec2.py --train_csv <path_to_train_csv> --val_csv <path_to_val_csv> --checkpoint_folder <path_to_store_checkpoint>```
    
    
where 

`--train_csv - Path to csv with details about training data`

`--val_csv - Path to csv with details about validation data`

`--checkpoint_folder - Place to store model checkpoints`


## Run Lyric Transcription for one file:
 
To perform lyric transcription for one file, run:

```python perform_lyrics_transcription_one_file.py --checkpoint_path <path_to_model_checkpoint> --lm_model_file <path_to_lm_model> --audio_path <path_to_audio_file>```    
    
where 

`--checkpoint_path - Path to csv with details about training data`

`--lm_model_file - Path to csv with details about validation data`

`--audio_path - Path to audio file to transcribe`

The script runs the trained Wav2Vec2 model  on the file and prints the transcribed lyrics. 
The transcribed lyrics are also stored in a file called **"transcribed_output.txt"**
