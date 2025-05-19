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


