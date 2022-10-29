import torch
import tempfile
from gevent.pywsgi import WSGIServer
import librosa
import soundfile as sf

import warnings

warnings.filterwarnings('ignore')

import sys

# Change system path to base directory
sys.path.append("..")

from melody_extraction.main import AST_Model
from melody_extraction.inference import predict_one_song
from speechbrain.pretrained import EncoderDecoderASR
from flask import Flask, jsonify, request, flash, redirect

app = Flask(__name__)

MELODY_MODEL_PATH = '../melody_extraction/results/best_model'
MELODY_FILE_PATH = 'upload.mp3'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_model = AST_Model(device, MELODY_MODEL_PATH)

asr_model = EncoderDecoderASR.from_hparams(
    source="trained_model",
    hparams_file="hyperparams.yaml",
    savedir="pretrained_model"
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'mp3'


@app.route('/melody', methods=['GET', 'POST'])
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
            file.save(MELODY_FILE_PATH)
            results = {}
            results = predict_one_song(best_model, MELODY_FILE_PATH, 'result', results, tomidi=False, output_path='',
                                       onset_thres=0.4, offset_thres=0.5)
            return jsonify(results)
    return '''
        <html>
            <body>
                <form action = "http://localhost:5000/melody" method = "POST" 
                    enctype = "multipart/form-data">
                    <input type = "file" name = "file" />
                    <input type = "submit"/>
                </form>   
            </body>
        </html>
    '''


@app.route('/lyrics', methods=['GET', 'POST'])
def transcribe_lyrics():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No selected file')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            extension = file.filename.split(".")[-1]
            handle, filename = tempfile.mkstemp()
            new_filename = filename + "." + extension
            file.save(new_filename)
            y, sr = librosa.load(new_filename)
            if sr != 16000:
                sf.write(new_filename, y, 16000)
            transcription = asr_model.transcribe_file(new_filename)
            file.close()
            payload = {"text": transcription}
            return jsonify(payload)
    return '''
        <html>
            <body>
                <form action = "http://localhost:5000/lyrics" method = "POST" 
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
