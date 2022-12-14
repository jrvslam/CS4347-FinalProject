import os
import warnings

import soundfile as sf
import torch
from datasets import Dataset, Audio
from flask import Flask, jsonify, request, flash, redirect
from flask_cors import CORS, cross_origin
from gevent.pywsgi import WSGIServer
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from transformers import AutomaticSpeechRecognitionPipeline
from pyctcdecode import build_ctcdecoder

from melody.dataset import OneSong
from melody.main import Melody_Model

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MELODY_MODEL_PATH = '../melody_extraction/results/best_model'
FILE_DUMP_PATH = "temp_folder"

processor = Wav2Vec2Processor.from_pretrained("checkpoint-4000")
lyrics_model = Wav2Vec2ForCTC.from_pretrained(
    "checkpoint-4000",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

# build the decoder
decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path="checkpoint-4000/3gram.arpa",
)

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)

pipeline = AutomaticSpeechRecognitionPipeline(model=lyrics_model, decoder = decoder,
                                              feature_extractor = processor.feature_extractor,
                                              tokenizer = processor.tokenizer,
                                              chunk_length_s=6)

melody_model = Melody_Model(device, MELODY_MODEL_PATH)
print("Models loaded!")

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

if not os.path.exists(FILE_DUMP_PATH):
    os.mkdir(FILE_DUMP_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['mp3', 'wav']

@app.route('/melody', methods=['GET', 'POST'])
@cross_origin()
def predict_melody():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No selected file')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            extension = file.filename.split(".")[-1]
            new_filename = os.path.join(FILE_DUMP_PATH, "temp" + "." + extension)
            y, sr = sf.read(file.stream)
            sf.write(new_filename, y, sr)
            test_dataset = OneSong(new_filename, 'result')
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                pin_memory=False,
                shuffle=False,
                drop_last=False,
            )
            results = melody_model.predict(test_loader)
            file.close()
            return jsonify(results)
    return '''
        <html>
            <body>
                <form action = "http://127.0.0.1:5000/melody" method = "POST" 
                    enctype = "multipart/form-data">
                    <input type = "file" name = "file" />
                    <input type = "submit"/>
                </form>   
            </body>
        </html>
    '''


@app.route('/lyrics', methods=['GET', 'POST'])
@cross_origin()
def transcribe_lyrics():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No selected file')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            extension = file.filename.split(".")[-1]
            new_filename = os.path.join(FILE_DUMP_PATH, "temp" + "." + extension)
            y, sr = sf.read(file.stream)
            sf.write(new_filename, y, sr)
            transcription = pipeline(new_filename)
            file.close()
            return jsonify(transcription)
    return '''
        <html>
            <body>
                <form action = "http://127.0.0.1:5000/lyrics" method = "POST" 
                    enctype = "multipart/form-data">
                    <input type = "file" name = "file" />
                    <input type = "submit"/>
                </form>   
            </body>
        </html>
    '''


if __name__ == '__main__':
    app.secret_key = 'hTWWG8UKwqt-PjWNkkAwkA'
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
