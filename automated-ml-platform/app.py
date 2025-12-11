# app.py
import os
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from ml_core import run_full_pipeline

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'csv'}
API_KEY = os.getenv('API_KEY', '')  # set in environment for simple protection

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(save_path)
        return jsonify({'message': 'File uploaded', 'filename': filename}), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/train', methods=['POST'])
def train():
    # Optional API key enforcement
    body = request.get_json() or {}
    header_key = request.headers.get('x-api-key', '')
    # Accept either in header or body
    provided = body.get('api_key') or header_key
    if API_KEY and provided != API_KEY:
        return jsonify({'error': 'Unauthorized - invalid API key'}), 401

    filename = body.get('filename')
    if not filename:
        return jsonify({'error': 'filename missing in request'}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'uploaded file not found on server'}), 400

    # options
    sequence_mode = body.get('sequence_mode', False)
    sequence_length = int(body.get('sequence_length', 10))
    target_col = body.get('target_col', 'target')

    try:
        # run_full_pipeline returns structured JSON-like dict
        results = run_full_pipeline(
            file_path,
            target_col=target_col,
            sequence_mode=sequence_mode,
            sequence_length=sequence_length,
            output_dir='static/plots'
        )
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# for convenience: serve plots
@app.route('/static/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory('static/plots', filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
