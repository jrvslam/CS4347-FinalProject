import torch
import torch.nn as nn
import torch.nn.functional as F

class MelodyCNN(nn.Module):
    '''
        CNN model used for melody extraction.
    '''
    def __init__(self, pitch_class=12, pitch_octave=4):
        super(MelodyCNN, self).__init__()
        self.pitch_octave = pitch_octave
        self.pitch_class = pitch_class
        
        torch.hub.list('rwightman/gen-efficientnet-pytorch')
        self.effnet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=False)
        
        self.effnet.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        num_ftrs = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Linear(num_ftrs, 2+pitch_class+pitch_octave+2)

    def forward(self, x):
        out = self.effnet(x)

        onset_logits = out[:, 0]
        offset_logits = out[:, 1]

        pitch_out = out[:, 2:]
        
        pitch_octave_logits = pitch_out[:, 0:self.pitch_octave+1]
        pitch_class_logits = pitch_out[:, self.pitch_octave+1:]

        
        return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits

