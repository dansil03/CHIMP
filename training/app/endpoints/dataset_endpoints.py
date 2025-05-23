import os
import shutil
import re
from io import BytesIO
from datetime import datetime
from flask import Blueprint, current_app, request, Request
from tempfile import mkdtemp
from werkzeug.exceptions import BadRequest
from zipfile import ZipFile, BadZipFile

bp = Blueprint("dataset", __name__)


@bp.route("/datasets")
def get_datasets():
    datastore = current_app.extensions["datastore"]
    return {
        "status": "successfully retrieved datasets",
        "datasets": [
            ds.replace("/", "")
            for ds in datastore.list_from_datastore("", recursive=False)
        ],
    }


@bp.route("/datasets", methods=["POST"])
def upload_dataset(passed_request: Request = None):
    current_request = request
    if passed_request:
        current_request = passed_request  # pragma: no cover

    if "file" not in current_request.files:
        raise BadRequest("No file in request")
    file = current_request.files["file"]

    print(current_request.files)
    print(file.filename)
    print(file.filename.endswith(".zip"))

    if not file.filename.endswith(".zip"):
        raise BadRequest("File should be a zip")

    dataset_name = current_request.form.get("dataset_name")
    if not dataset_name:
        raise BadRequest("Dataset name ('dataset_name') field missing")
    invalid_chars = re.compile(r'[<>:"/\\|?*]')
    if invalid_chars.search(dataset_name):
        raise BadRequest(
            "Dataset name ('dataset_name') should only contain characters allowed in path strings"
        )

    datastore = current_app.extensions["datastore"]
    if dataset_name in [
        ds.replace("/", "") for ds in datastore.list_from_datastore("", recursive=False)
    ]:
        raise BadRequest(f"Dataset with name '{dataset_name}' already exists")

    tmpdir = mkdtemp(prefix="chimp_")
    zip_path = os.path.join(tmpdir, file.filename)
    file.save(zip_path)
    upload_path = os.path.join(tmpdir, "to_upload")
    os.mkdir(upload_path)

    try:
        with ZipFile(zip_path, "r") as f:
            f.extractall(upload_path)
    except BadZipFile:
        raise BadRequest("Invalid zip file")

    print("start uploading")
    datastore.store_file_or_folder(dataset_name, upload_path)
    print("done uploading")

    shutil.rmtree(tmpdir)
    return {"status": "successfully uploaded dataset"}


@bp.route("/datasets/<path:object_path>")
def get_dataset_file(object_path):
    datastore = current_app.extensions["datastore"]
    data = datastore.load_object_to_memory(object_path)

    if data is None:
        return {"error": "File not found"}, 404

    # Probeer te detecteren of het een JSON-bestand is
    if object_path.endswith(".json"):
        try:
            import json
            return json.loads(data.read())
        except Exception as e:
            return {"error": f"Invalid JSON: {str(e)}"}, 500

    # Anders gewoon raw bestand teruggeven (optioneel)
    return current_app.response_class(
        data.getvalue(), mimetype="application/octet-stream"
    )



@bp.route("/label_image", methods=["POST"])
def label_image():
    dataset = request.form.get("dataset_name")
    emotion = request.form.get("emotion")
    file = request.files.get("file")

    if not all([dataset, emotion, file]):
        return {"error": "Missing data"}, 400

    if file.filename == "":
        return {"error": "No file selected"}, 400

    from flask import current_app
    datastore = current_app.extensions["datastore"]

    label_path = f"{dataset}/training/{emotion}/{file.filename}"
    buffer = BytesIO(file.read())
    buffer.seek(0)

    datastore.store_object(label_path, buffer, file.filename)

    return {"status": "label saved", "path": label_path}

