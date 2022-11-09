# Melody Extraction

This folder contains the code used for training and evaluating the melody extraction model.

## Data used for model training

A subset of the MIR-ST500 dataset is used to train the melody transcription model.
To train the model with minimal file path configuration, the data should be placed in this folder as follows:

```
data
|- test
|- train
|- valid
|- annotations.json
```

The exact training data used were as follows:
- `train`: 1-366 (some song IDs missing, total 350 songs used)
- `valid`: 401-440
- `test`: 441-450
- Total: 400 songs

## Training the model:
 
To prepare the datasets, run:
```
python dataset.py --data_dir ./data --annotation_path ./data/annotations.json --save_dataset_dir ./data/
```

To train the model, run:
```
python main.py --train_dataset_path ./data/train.pkl --valid_dataset_path ./data/valid.pkl --save_model_dir ./results
```
The [EffectiveNet](https://github.com/rwightman/gen-efficientnet-pytorch) is used for the training of the final model. Training of 1 epoch is expected to take 2-3 hours.


## Evaluating the model:

To measure the model's performance on the test dataset, run the following:
```
python inference.py
python evaluate.py
```

## Run melody extraction for one file:
 
To perform melody extraction on one file, run:

```
python melody_extraction_one_file.py --file_path <path_to_file> --output_path <path_to_output_dir> --to_midi <True/False>
```

The script will run melody extraction on the given file and save the prediction annotations in the specified directory.
If `to_midi` is set to `True`, the MIDI file will also be saved in the specified directory.


