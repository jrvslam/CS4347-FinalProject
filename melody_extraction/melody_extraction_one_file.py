import torch

import os
import json
import argparse
from main import Melody_Model
from inference import predict_one_song


if __name__ == '__main__':
    '''
    This script performs inference on one file using the trained singing transcription model in main.py.
    
    Sample usage:
    python melody_extraction_one_file.py --file_path <path_to_file> --save_dir <path_to_save_dir> --to_midi <True/False>
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='path to the input audio')
    parser.add_argument('--output_path', default='./predictions', help='path to the output prediction folder')
    parser.add_argument('--to_midi', default=False, help='whether to save the extracted melody as a MIDI file')
    parser.add_argument('--save_model_dir', default='./results', help='path to the trained model')
    parser.add_argument('--onset_thres', default=0.4, help='onset threshold')
    parser.add_argument('--offset_thres', default=0.5, help='offset threshold')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_model_path = args.save_model_dir + '/best_model'
    best_model = Melody_Model(device, best_model_path)
    output_json_path = args.output_path + '/predictions.json'

    results = {}
    
    results = predict_one_song(best_model, args.file_path, 'result', results, tomidi=args.to_midi, output_path=os.path.join(args.output_path, 'trans.mid'), 
                            onset_thres=float(args.onset_thres), offset_thres=float(args.offset_thres))

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    with open(output_json_path, 'w') as f:
        output_string = json.dumps(results)
        f.write(output_string)

    