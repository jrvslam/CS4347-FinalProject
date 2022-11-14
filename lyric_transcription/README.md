# Lyric Transcription

This folder provides the code resources for training the Lyric Transcription Model, obtain the WER scores on the test sets and perform inference on a single file.


## Prepare the files for training the model:


    The Wav2Vec2 Training scripts require a csv with the following columns:

    a) text: The transcribed lyrics of a given audio file

    b) file: The path to the audio file

    c) audio: The path to the audio file again (internally, this is used by the module to load the audio file)


## Training the Wav2Vec2 model:
 
To train the Wav2Vec2 model, run:

    
    python train_wav2vec2.py --train_csv <path_to_train_csv> --val_csv <path_to_val_csv> --checkpoint_folder <path_to_store_checkpoint>
    
    
where 

`--train_csv - Path to csv with details about training data`

`--val_csv - Path to csv with details about validation data`

`--checkpoint_folder - Place to store model checkpoints`


## Training the n-gram Language Model:

To train an n-gram language model, we use the [KenLM package](https://github.com/kpu/kenlm).
Please refer to the instructions provided in the KenLM repository for setting up the package and building an n-gram language model.
It is recommended to take the transcriptions from the training corpus, and dump each transcription in a new-line separated text file, and use that with KenLM for training the language model.

## Evaluating the Wav2Vec2 model:
 
To evaluate the Wav2Vec2 model, run:

    
    python obtain_wer_scores_test_set.py --val_csv <path_to_val_csv> --dsing_test_csv <path_to_dsing_test_csv> --nem_test_csv <path_to_nem_csv> --checkpoint_folder <path_to_model_checkpoint>
    
    
where 

`--val_csv - Path to csv with details about validation data`

`--dsing_test_csv - Path to csv with details about DSing test data`

`--nem_test_csv - Path to csv with details about N20EM test data`

`--checkpoint_folder - Place to load model checkpoint from`

The scores are saved in a csv folder named **wav2vec2_only_scores.csv**.

## Evaluating the Wav2Vec2 model with n-gram LM:
 
To evaluate the Wav2Vec2 model with an n-gram LM, run:

    
    python obtain_wer_scores_test_set_with_lm.py --val_csv <path_to_val_csv> --dsing_test_csv <path_to_dsing_test_csv> --nem_test_csv <path_to_nem_csv> --checkpoint_folder <path_to_model_checkpoint> --lm_weights <path_to_n_gram_arpa_file>
    
    
where 

`--val_csv - Path to csv with details about validation data`

`--dsing_test_csv - Path to csv with details about DSing test data`

`--nem_test_csv - Path to csv with details about N20EM test data`

`--checkpoint_folder - Place to load model checkpoint from`

`--lm_weights - Place to load n-gram LM from`

The scores are saved in a csv folder named **wav2vec2_only_scores_lm.csv**.

## Run Lyric Transcription for one file:
 
To perform lyric transcription for one file, run:

    python perform_lyrics_transcription_one_file.py --checkpoint_path <path_to_model_checkpoint> --lm_model_file <path_to_lm_model> --audio_path <path_to_audio_file>    
    
where 

`--checkpoint_path - Path to checkpoint folder (checkpoint-4000 folder)`

`--lm_model_file - Path to .arpa folder present inside the model checkpoint folder (checkpoint-4000/3gram.arpa)`

`--audio_path - Path to audio file to transcribe`

The script runs the trained Wav2Vec2 model  on the file and prints the transcribed lyrics. 
The transcribed lyrics are also stored in a file called **"transcribed_output.txt"**


## Training the Whisper model:
 
To train the Whisper-Tiny model, run:

    
    python train_whisper.py --train_csv <path_to_train_csv> --val_csv <path_to_val_csv> --checkpoint_folder <path_to_store_checkpoint>
    
    
where 

`--train_csv - Path to csv with details about training data`

`--val_csv - Path to csv with details about validation data`

`--checkpoint_folder - Place to store model checkpoints`

## Evaluating the Whisper model:
 
To evaluate the Whisper-Tiny model, run:

    
    python obtain_wer_scores_test_set_whisper.py --val_csv <path_to_val_csv> --dsing_test_csv <path_to_dsing_test_csv> --nem_test_csv <path_to_nem_csv> --checkpoint_folder <path_to_model_checkpoint>
    
    
where 

`--val_csv - Path to csv with details about validation data`

`--dsing_test_csv - Path to csv with details about DSing test data`

`--nem_test_csv - Path to csv with details about N20EM test data`

`--checkpoint_folder - Place to load model checkpoint from`

The scores are saved in a csv folder named **Results_whisper.csv**.

# References and Resources

The following links and blogs were extremely useful in terms of understanding the various concepts, and utilizing code to train our models with on the new lyric dataset.

* [Fine-Tune Wav2Vec2 for English ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-wav2vec2-english)
* [Boosting Wav2Vec2 with n-grams in 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram)
* [Using N-Gram LMs with Wav2Vec2](http://mohitmayank.com/a_lazy_data_science_guide/audio_intelligence/wav2vec2/)
* [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)


