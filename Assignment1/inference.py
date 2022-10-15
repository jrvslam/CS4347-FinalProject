import torch
from torch.utils.data import DataLoader

import os
import json
import argparse
import mido
from tqdm import tqdm
from pathlib import Path
from main import AST_Model
from dataset import OneSong

import warnings
warnings.filterwarnings('ignore')

def notes2mid(notes):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480
    new_tempo = mido.bpm2tempo(120.0)

    track.append(mido.MetaMessage('set_tempo', tempo=new_tempo))
    track.append(mido.Message('program_change', program=0, time=0))

    cur_total_tick = 0

    for note in notes:
        if note[2] == 0:
            continue
        note[2] = int(round(note[2]))

        ticks_since_previous_onset = int(mido.second2tick(note[0], ticks_per_beat=480, tempo=new_tempo))
        ticks_current_note = int(mido.second2tick(note[1]-0.0001, ticks_per_beat=480, tempo=new_tempo))
        note_on_length = ticks_since_previous_onset - cur_total_tick
        note_off_length = ticks_current_note - note_on_length - cur_total_tick

        track.append(mido.Message('note_on', note=note[2], velocity=100, time=note_on_length))
        track.append(mido.Message('note_off', note=note[2], velocity=100, time=note_off_length))
        cur_total_tick = cur_total_tick + note_on_length + note_off_length

    return mid
    

def convert_to_midi(predicted_result, song_id, output_path):
    to_convert = predicted_result[song_id]
    mid = notes2mid(to_convert)
    mid.save(output_path)


def predict_one_song(model, input_path, song_id, results, tomidi, output_path, onset_thres, offset_thres):
    test_dataset = OneSong(input_path, song_id)
    test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )
    results = model.predict(test_loader, results=results, onset_thres=onset_thres, offset_thres=offset_thres)
    
    if tomidi:
        convert_to_midi(results, song_id, output_path)

    return results


def predict_whole_dir(model, test_dir, results, output_json_path, onset_thres, offset_thres):
    results = {}

    for song_dir in tqdm(os.listdir(test_dir)):
        
        input_path = os.path.join(test_dir, song_dir, "Mixture.mp3")
        song_id = song_dir

        results = predict_one_song(model, input_path, song_id, results, tomidi=True, output_path=os.path.join(test_dir, song_dir, "trans.mid"), 
                onset_thres=onset_thres, offset_thres=offset_thres)

    with open(output_json_path, 'w') as f:
        output_string = json.dumps(results)
        f.write(output_string)
    
    return results


def make_predictions(testset_path, output_path, model, onset_thres, offset_thres, song_id='1'):
    results = {}
    if os.path.isfile(testset_path):
        results = predict_one_song(model, testset_path, song_id, results, tomidi=True, output_path=output_path, 
                            onset_thres=float(onset_thres), offset_thres=float(offset_thres))

    elif os.path.isdir(testset_path):
        results = predict_whole_dir(model, testset_path, results, output_json_path=output_path, 
                        onset_thres=float(onset_thres), offset_thres=float(offset_thres))

    else:
        print ("\"input\" argument is not valid")

    return results


if __name__ == '__main__':
    """
    This script performs inference using the trained singing transcription model in main.py.
    
    Sample usage:
    python inference.py --best_model_id 9 
    The best model may not be number 9. It depends on your result of validation.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset_path", default='./data/test', help="path to the input audio/folder")
    parser.add_argument('--output_path', default='./data/predictions.json', help="path to the output prediction json")
    parser.add_argument('--save_model_dir', default='./results', help='path to the trained model')
    parser.add_argument("--best_model_id", help='best model id got in the training and validation')
    parser.add_argument("--onset_thres", default=0.4, help="onset threshold")
    parser.add_argument("--offset_thres", default=0.5, help="offset threshold")
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    best_model_path = args.save_model_dir + '/model_' + str(args.best_model_id)
    best_model = AST_Model(device, best_model_path)
    
    make_predictions(args.test_dataset_path, args.output_path, best_model, args.onset_thres, args.offset_thres)

    