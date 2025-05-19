import os
import numpy as np
import cv2
from os import environ
import logging
import requests
import json
import zipfile
import re
from PIL import Image

from flask_socketio import SocketIO, emit
from flask import request
from datetime import datetime
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from logic.image_processor import ImageProcessor
from io import BytesIO

import imageio.v3 as iio

INFERENCE_INTERVAL = 0

_logger = logging.getLogger(environ.get('logger-name', 'chimp-ml-frontend'))
_image_processors: dict = {}


def _on_connect():
    _logger.debug(f'Web client connected: {request.sid}')
    _image_processors[request.sid] = ImageProcessor(INFERENCE_INTERVAL)


def _on_disconnect():
    _logger.debug(f'Web client disconnected: {request.sid}')
    print(request)
    # Note: has a vulnerability in which by not explicitly disconnecting from other side of the socket, more image
    #           processors keep getting cached
    if request.sid in _image_processors:
        del _image_processors[request.sid]


def _process_image(data):
    user_id = data['user_id'] if data['user_id'] != '' else request.sid
    image_blob = data['image_blob']

    img_processor = _image_processors.get(user_id, ImageProcessor(INFERENCE_INTERVAL))
    img_processor.load_image(image_blob)
    img_processor.process(user_id)

    data_to_emit = {'predictions': img_processor.predictions, 'status': img_processor.status_msg}

    emit('update-data', data_to_emit)

    return img_processor.get_image_blob()

def sanitize_timestamp(timestamp):
    return timestamp.replace("T", "_").replace(":", "-").replace(".", "-")

def _process_video(data):
    print("[START] Processing video blobs")
    
    # Load the face detection cascade file
    cascade_file = os.path.join(os.getcwd(), 'static', 'cascades', 'frontalface_default_haarcascade.xml')
    face_cascade = cv2.CascadeClassifier(cascade_file)

    # Retrieve environment variables and construct the dataset upload URL
    EXPERIMENT_NAME = environ.get("EXPERIMENT_NAME")
    #PLUGIN_NAME = "Emotion+Recognition" if not is_pool else "Active+Learning"
    TRAINING_SERVER_URL = environ.get("TRAINING_SERVER_URL")
    url = TRAINING_SERVER_URL + "/datasets"

    # Extract data from the incoming request
    user_id = data['user_id'] if data['user_id'] != '' else request.sid
    username = data['username']
    video_blobs = data['image_blobs']
    emotions = data['emotions']
    timestamps = data['timestamps'] if 'timestamps' in data else []
    is_pool = data.get('is_pool', False)

    print(f"Incoming data | User ID: {user_id}, Username: {username}, Is Pool: {is_pool}")
    print(f"Number of video blobs: {len(video_blobs)} | Emotions: {emotions} | Timestamps: {len(timestamps)}")

    # Generate default timestamps if missing or mismatched
    if not timestamps or len(timestamps) != len(video_blobs):
        print("Timestamps are missing or do not match the number of video blobs. Generating default timestamps.")
        timestamps = [f"{datetime.utcnow().isoformat()}-{i}" for i in range(len(video_blobs))]

    # Create a unique ID for the calibration dataset
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    unique_id = f"{username}_{timestamp}_{user_id}"
    clean_id = f"calibration_{unique_id}"
    zip_buffer = BytesIO()
    zip_buffer.name = clean_id

    total_frames_detected = 0
    total_images_written = 0
 
    # Process each video blob and extract face images
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        global_frame_counter = 0 
        for blob_index, (video_blob, emotion, ts) in enumerate(zip(video_blobs, emotions, timestamps)):
            blob_size = len(video_blob)
            print(f"\n[INFO] Processing blob {blob_index+1}/{len(video_blobs)} | Emotion: {emotion} | Size: {blob_size} bytes | Timestamp: {ts}")
            if blob_size < 1000:
                print(f"[WARNING] Blob {blob_index+1} is very small — possible recording issue.")

            # Sanitize the timestamp for use in filenames
            ts = sanitize_timestamp(ts)
            video_stream = BytesIO(video_blob)
            video_stream.seek(0)

            try:
                # Read the video blob into a frame array
                video_array = iio.imread(video_stream, plugin='pyav')
                print(f"[INFO] Loaded {len(video_array)} frames from blob.")
            except Exception as e:
                print(f"[ERROR] Failed to read video blob {blob_index+1}: {e}")
                continue

            # Process each frame in the video
            for i, img in enumerate(video_array):
                try:
                    # Convert the frame to grayscale
                    grey_frame = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                except Exception as e:
                    print(f"[WARNING] Failed to convert frame {i} to grayscale: {e}")
                    continue

                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(grey_frame, 1.3, 5)
                print(f"[DEBUG] Frame {i}: {len(faces)} face(s) detected")

                # Skip frames with no or multiple faces
                if len(faces) != 1:
                    print(f"[SKIP] Frame {i}: Skipping due to face count != 1")
                    continue

                # Process each detected face
                for index, (x, y, width, height) in enumerate(faces):
                    try:
                        # Crop, resize, and save the face image
                        image = cv2.resize(grey_frame[y:y+height, x:x+width], (96, 96))
                        image = Image.fromarray(image.astype('uint8'))
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        buffer.seek(0)

                        # Create a unique filename for the image
                        if is_pool:
                            name = f'img_pool_{global_frame_counter:05d}.png'
                            zip_path = os.path.join("pool", name)
                        else:
                            name = f'img_{emotion}_{global_frame_counter:05d}.png'
                            zip_path = os.path.join("train", emotion, name)

                        global_frame_counter += 1


                        # Write the image to the zip file
                        zipf.writestr(zip_path, buffer.getvalue())
                        total_images_written += 1
                        print(f"[WRITE] Wrote image to zip: {zip_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to process face crop in frame {i}: {e}")

                total_frames_detected += 1

    # Log summary of processing
    zip_size = zip_buffer.getbuffer().nbytes
    print(f"\n[SUMMARY] Total frames processed with faces: {total_frames_detected}")
    print(f"[SUMMARY] Total images written to zip: {total_images_written}")
    print(f"[SUMMARY] Final ZIP file size: {zip_size} bytes")

    # Prepare the zip file for upload
    zip_buffer.seek(0)
    files = {"file": (clean_id + '.zip', zip_buffer.getvalue(), 'application/zip')}

    print("[UPLOAD] Sending zip to dataset_name:", clean_id)
    try:
        # Upload the zip file to the server
        response = requests.request('POST', url=url, data={"dataset_name": clean_id}, files=files)
        print(f"[UPLOAD RESPONSE] Status {response.status_code}")
        print(f"[UPLOAD RESPONSE] Body: {response.json()}")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return {"error": str(e)}, 500

    # Handle upload errors
    if response.status_code != 200:
        return response.json(), response.status_code

    # Prepare dataset paths for calibration
    datasets = {}
    if is_pool:
       # datasets["pool"] = f"{clean_id}/pool"
       datasets["pool"] = clean_id
    else:
        datasets["train"] = f"{clean_id}/train"

    PLUGIN_NAME = "Emotion+Recognition" if not is_pool else "Active+Learning"


    
    # Prepare the form data for calibration task
    form = {
        "calibration_id": f"{username}_{user_id}",  # Unique calibration identifier
        "calibrate": True,                          # Indicate calibration mode
        "experiment_name": EXPERIMENT_NAME,         # Experiment name from environment
        "datasets": json.dumps(datasets)            # Datasets paths as JSON string
    }

    print(f"[TASK] Sending task request to plugin '{PLUGIN_NAME}' with form data:", form)
    # Send POST request to trigger calibration task on the training server
    response = requests.request('POST', url=TRAINING_SERVER_URL + "/tasks/run/" + PLUGIN_NAME, data=form)
    print(f"[TASK RESPONSE] Status {response.status_code}: {response.json()}")

    # Return the response from the training server
    return response.json(), response.status_code


def _train():
    PLUGIN_NAME="Emotion+Recognition"
    
    EXPERIMENT_NAME=environ.get("EXPERIMENT_NAME")
    TRAINING_SERVER_URL=environ.get("TRAINING_SERVER_URL")
    datasets=json.dumps({"train": "emotions"})
    url = TRAINING_SERVER_URL + "/tasks/run/" + PLUGIN_NAME

    response = requests.request('POST',  url=url, data={"datasets" : datasets, "experiment_name" : EXPERIMENT_NAME})

    return response.json(), response.status_code

def _calibrate():
    PLUGIN_NAME="Emotion+Recognition"
    
    EXPERIMENT_NAME=environ.get("EXPERIMENT_NAME")
    TRAINING_SERVER_URL=environ.get("TRAINING_SERVER_URL")
    
    url = TRAINING_SERVER_URL + "/tasks/run/" + PLUGIN_NAME

    form = dict()
    files = {}

    # get user_id from request
    if "user_id" not in request.args:
        return BadRequest("No user specified.")
    
    user_id = request.args["user_id"]
    form["calibration_id"] = user_id
    form["calibrate"] = True
    form["experiment_name"] = EXPERIMENT_NAME

    # get zipfile from request
    if len(request.files) == 0:
        return BadRequest("No files uploaded.")
    if "zipfile" not in request.files:
        return BadRequest("Different file expected.")
    file = request.files["zipfile"]
    if file.filename == "":
        return BadRequest("No file selected.")
    if not (
        "." in file.filename and file.filename.rsplit(".", 1)[1].lower() == "zip"
    ):
        return BadRequest("File type not allowed. Must be a zip.")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    calibration_dataset_name = secure_filename(
        f"calib_emotions{user_id}{timestamp}".replace("-", "").replace("_", "")
    )
    #convert to bytes?
    file_bytes = BytesIO()
    file.save(file_bytes)
    file_bytes.seek(0)
    file_bytes.name=file.filename
    files["file"] = (file_bytes.name, file_bytes, 'application/zip')
    
    form["datasets"] = json.dumps({"train": "emotions", "calibration" : calibration_dataset_name})
    print(calibration_dataset_name)
    url = TRAINING_SERVER_URL + "/datasets"
    response = requests.request('POST',  url=url, data={"dataset_name" : calibration_dataset_name}, files=files)
    if response.status_code!=200:
        return response.json(), response.status_code
        #raise BadRequest("Could not upload dataset zip")
        
    url = TRAINING_SERVER_URL + "/tasks/run/" + PLUGIN_NAME
    response = requests.request('POST',  url=url, data=form)

    return response.json(), response.status_code


def add_as_websocket_handler(socket_io: SocketIO, app):
    global _on_connect, _on_disconnect, _process_image, _process_video

    _on_connect = socket_io.on('connect')(_on_connect)
    _on_disconnect = socket_io.on('disconnect')(_on_disconnect)
    _process_video = socket_io.on('process-video')(_process_video)
    _process_image = socket_io.on('process-image')(_process_image)

    app.route('/train', methods=['POST'])(_train)
    app.route('/calibrate', methods=['POST'])(_calibrate)

    return socket_io