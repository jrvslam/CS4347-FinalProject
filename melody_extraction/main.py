import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import pickle
import argparse
import librosa
import numpy as np
from tqdm import tqdm

from model import MelodyCNN
from dataset import SingingDataset
from pitch_correction import perform_pitch_correction

# Set seed for reproducability
torch.manual_seed(0)

FRAME_LENGTH = librosa.frames_to_time(1, sr=44100, hop_length=1024)

class Melody_Model:
    '''
        This is main class for training model and making predictions.
    '''
    def __init__(self, device= "cuda:0", model_path=None):
        # Initialize model
        self.device = device 
        self.model = MelodyCNN().to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded.')
        else:
            print('Model initialized.')

    def fit(self, args, learning_params):
        # Set paths
        trainset_path = args.train_dataset_path
        validset_path = args.valid_dataset_path
        save_model_dir = args.save_model_dir
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)

        # Load dataset
        print('Loading datasets...')
        with open(trainset_path, 'rb') as f:
            trainset = pickle.load(f)
        with open(validset_path, 'rb') as f:
            validset = pickle.load(f)

        trainset_loader = DataLoader(
            trainset,
            batch_size=learning_params['batch_size'],
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        validset_loader = DataLoader(
            validset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        # Set optimizer and loss functions
        optimizer = optim.Adam(self.model.parameters(), lr=learning_params['lr'])
        onset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0,], device=device))
        offset_criterion = nn.BCEWithLogitsLoss()
        octave_criterion = nn.CrossEntropyLoss(ignore_index=100)
        pitch_criterion = nn.CrossEntropyLoss(ignore_index=100)

        start_time = time.time()
        best_model_id = -1
        min_valid_loss = 10000
        epoch_num = learning_params['epoch']

        # Start training 
        print('Start training...')

        for epoch in range(epoch_num):
            self.model.train()

            for inputs, labels in trainset_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                onset_output, offset_output, pitch_octave_output, pitch_class_output = self.model(inputs)

                # Loss computing
                onset_loss = onset_criterion(onset_output, labels[:, 0])
                offset_loss = offset_criterion(offset_output, labels[:, 1])
                pitch_octave_loss = octave_criterion(pitch_octave_output, labels[:, 2].long())
                pitch_class_loss = pitch_criterion(pitch_class_output, labels[:, 3].long())

                # Backward pass
                optimizer.zero_grad()
                onset_loss.backward(retain_graph=True)
                offset_loss.backward(retain_graph=True)
                pitch_octave_loss.backward(retain_graph=True)
                pitch_class_loss.backward()
                optimizer.step()

            # Vaidate
            running_loss = 0
            running_split_loss = np.zeros(4)
            num_batches = 0

            self.model.eval()
            for inputs, labels in validset_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                onset_output, offset_output, pitch_octave_output, pitch_class_output = self.model(inputs)

                # Loss computing
                onset_loss = onset_criterion(onset_output, labels[:, 0])
                offset_loss = offset_criterion(offset_output, labels[:, 1])
                pitch_octave_loss = octave_criterion(pitch_octave_output, labels[:, 2].long())
                pitch_class_loss = pitch_criterion(pitch_class_output, labels[:, 3].long())

                running_split_loss = np.add(running_split_loss, [float(onset_loss), float(offset_loss), float(pitch_octave_loss), float(pitch_class_loss)])

                num_batches += 1

            running_loss = np.average(running_split_loss)

            total_loss = running_loss / num_batches
            total_onset_loss = running_split_loss[0] / num_batches
            total_offset_loss = running_split_loss[1] / num_batches
            total_pitch_octave_loss = running_split_loss[2] / num_batches
            total_pitch_class_loss = running_split_loss[3] / num_batches

            print('epoch=', epoch + 1)
            print('total loss=', total_loss)
            print('total onset loss=', total_onset_loss)
            print('total offset loss=', total_offset_loss)
            print('total pitch octave loss=', total_pitch_octave_loss)
            print('total pitch class loss=', total_pitch_class_loss, '\n')

            if (total_loss < min_valid_loss):
                best_model_id = epoch + 1
                min_valid_loss = total_loss
                torch.save(self.model.state_dict(), save_model_dir + '/best_model')
                
        print('Training done in {:.1f} minutes.'.format((time.time()-start_time)/60))
        return best_model_id


    def parse_frame_info(self, frame_info, onset_thres, offset_thres):
        """Parse frame info [(onset_probs, offset_probs, note number)...] into desired label format."""

        result = []
        current_onset = None
        pitch_counter = []
        local_max_size = 3
        current_time = 0.0

        onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])
        onset_seq_length = len(onset_seq)

        for i in range(len(frame_info)):

            current_time = FRAME_LENGTH*i
            info = frame_info[i]

            backward_frames = i - local_max_size
            if backward_frames < 0:
                backward_frames = 0

            forward_frames = i + local_max_size + 1
            if forward_frames > onset_seq_length - 1:
                forward_frames = onset_seq_length - 1

            # local max and more than threshold
            if info[0] >= onset_thres and onset_seq[i] == np.amax(onset_seq[backward_frames : forward_frames]):

                if current_onset is None:
                    current_onset = current_time   
                else:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                    current_onset = current_time
                    pitch_counter = []

            # if it is offset
            elif info[1] >= offset_thres:  
                if current_onset is not None:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                    current_onset = None
                    pitch_counter = []

            # If current_onset exist, add count for the pitch
            if current_onset is not None:
                final_pitch = int(info[2]* 12 + info[3])
                if info[2] != 4 and info[3] != 12:
                    pitch_counter.append(final_pitch)

        if current_onset is not None:
            if len(pitch_counter) > 0:
                result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
            current_onset = None

        return result

    def predict(self, test_loader, results={}, onset_thres=0.4, offset_thres=0.5):
        """Predict results for a given test dataset."""

        # Start predicting
        self.model.eval()
        with torch.no_grad():
            song_frames_table = {}
            
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                # Parse batch data
                input_tensor = batch[0].to(self.device)
                song_ids = batch[1]

                result_tuple = self.model(input_tensor)
                onset_logits = result_tuple[0]
                offset_logits = result_tuple[1]
                pitch_octave_logits = result_tuple[2]
                pitch_class_logits = result_tuple[3]

                onset_probs, offset_probs = torch.sigmoid(onset_logits).cpu(), torch.sigmoid(offset_logits).cpu()
                pitch_octave_logits, pitch_class_logits = pitch_octave_logits.cpu(), pitch_class_logits.cpu()

                # Collect frames for corresponding songs
                for bid, song_id in enumerate(song_ids):
                    frame_info = (onset_probs[bid], offset_probs[bid], torch.argmax(pitch_octave_logits[bid]).item()
                            , torch.argmax(pitch_class_logits[bid]).item())

                    song_frames_table.setdefault(song_id, [])
                    song_frames_table[song_id].append(frame_info)    

            # Parse frame info into output format for every song
            for song_id, frame_info in song_frames_table.items():
                results[song_id] = perform_pitch_correction(self.parse_frame_info(frame_info, onset_thres=onset_thres, offset_thres=offset_thres))
        
        return results


if __name__ == '__main__':
    '''
    This script performs training and validation of the singing transcription model.
    
    Sample usage:
    python main.py --train_dataset_path ./data/train.pkl --valid_dataset_path ./data/valid.pkl --save_model_dir ./results
    or 
    python main.py (All parameters are defualt)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', default='./data/train.pkl', help='path to the train set')
    parser.add_argument('--valid_dataset_path', default='./data/valid.pkl', help='path to the valid set')
    parser.add_argument('--save_model_dir', default='./results', help='path to save the trained models')
    
    args = parser.parse_args()
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ast_model = Melody_Model(device)

    # Set learning params
    learning_params = {
        'batch_size': 50,
        'epoch': 5,
        'lr': 1e-4,
    }

    # Train and Validation
    best_model_id = ast_model.fit(args, learning_params)
    print('Best Epoch: ', best_model_id)
    