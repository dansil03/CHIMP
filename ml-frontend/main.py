import os
from os import environ
import sys
import json
from io import BytesIO
import requests

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO
from utils.logging_config import configure_logging
from request_handlers import inference_handler
from dotenv import load_dotenv

# Zorg dat imports uit parent directory werken
basedir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(basedir, "..")))

# Setup Flask app en socket
app = Flask(__name__)
app.secret_key = os.urandom(24)

socket_io = SocketIO(app, always_connect=True, logger=False, engineio_logger=False)
socket_io = inference_handler.add_as_websocket_handler(socket_io, app)
configure_logging(app)

# User database
users = {
    'user1': 'banaan',
    'user2': 'password2',
    'maarten': 'maarten',
    'eddy': 'eddy',
    'abdul': 'abdul',
    'silas': 'silas'
}


TRAINING_SERVER_URL = environ.get("TRAINING_SERVER_URL")

@app.route("/api/unlabeled_datasets")
def get_unlabeled_datasets():
    try:
        print("Requesting dataset list from:", f"{TRAINING_SERVER_URL}/datasets")
        response = requests.get(f"{TRAINING_SERVER_URL}/datasets")
        response.raise_for_status()
        datasets = response.json().get("datasets", [])
        print("Alle datasets:", datasets)

        selection_files = []
        datasets_info = []

        for dataset in datasets:
            selection_url = f"{TRAINING_SERVER_URL}/datasets/{dataset}/selection/selection.json"
            print(f"Trying selection file at: {selection_url}")
            resp = requests.get(selection_url)
            if resp.status_code == 200:
                try:
                    selection_json = resp.json()
                    selection_files.append((dataset, selection_json))
                except Exception as parse_error:
                    print(f"Fout bij parsen van JSON voor {dataset}: {parse_error}")
            else:
                print(f"Geen selection.json gevonden voor {dataset} (status {resp.status_code})")

        print("Gevonden selection.json bestanden:", [ds for ds, _ in selection_files])

        for dataset, selection_data in selection_files:
            print(f"Selection data voor {dataset}:", selection_data)

            user = extract_user_from_name(dataset)
            total = selection_data.get("total", 0)
            timestamp = selection_data.get("timestamp", "")[:16].replace("T", " ")
            labeled_percentage = 0  # pas aan als labels geteld worden

            datasets_info.append({
                "dataset_id": dataset,
                "user": user,
                "total_images": total,
                "labeled_percentage": labeled_percentage,
                "received": timestamp
            })

        datasets_info.sort(key=lambda x: x["received"], reverse=True)
        print("Final datasets_info:", datasets_info)

        return jsonify(datasets_info)

    except requests.RequestException as e:
        print("RequestException occurred:", str(e))
        return jsonify({"error": str(e)}), 500


from flask import make_response, Response

@app.route("/proxy_image/<path:filepath>")
def proxy_image(filepath):
    minio_url = f"{TRAINING_SERVER_URL}/datasets/{filepath}"
    print(f"[DEBUG] Proxying image request to: {minio_url}")

    resp = requests.get(minio_url)
    if resp.status_code != 200:
        return f"Image not found: {filepath}", 404

    
    content_type = resp.headers.get('Content-Type', 'image/png')  
    response = make_response(resp.content)
    response.headers.set('Content-Type', content_type)
    return response


@app.route("/proxy_label", methods=["POST"])
def proxy_label():
    dataset = request.form.get("dataset_name")
    emotion = request.form.get("emotion")
    file = request.files.get("file")

    if not all([dataset, emotion, file]):
        return jsonify({"error": "Missing data"}), 400

    files = {
        "file": (file.filename, file.read(), file.content_type)
    }
    data = {
        "dataset_name": dataset,
        "emotion": emotion
    }

    response = requests.post(f"{TRAINING_SERVER_URL}/label_image", data=data, files=files)

    if response.status_code != 200:
        print("Proxy labeling error:", response.text)
        return jsonify({"error": "Labeling failed"}), 500

    return response.json(), 200



def extract_user_from_name(dataset_name):
    """
    Exporteer 'silas' uit bijvoorbeeld 'calibration_silas_2025-05-22...'
    """
    parts = dataset_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


# Webpagina routes
@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/kali')
def kali_page():
    if 'username' in session:
        return render_template('kali.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/unlabeled_overview')
def unlabeled_overview_page():
    if 'username' in session:
        return render_template('unlabeled_overview.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/label')
def label_page():
    if 'username' in session:
        dataset = request.args.get("dataset")
        if not dataset:
            return "Dataset not specified", 400
        return render_template('label.html', username=session['username'], dataset=dataset)
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# App runner
def run_app():
    return socket_io.run(app=app, host='0.0.0.0', port=5252, debug=False)

def get_app():
    load_dotenv()
    return app

if __name__ == '__main__':
    load_dotenv()
    run_app()
