import os
import json
import numpy as np
import tensorflow as tf
import glob
import datetime
from typing import Dict, Optional, List
from tensorflow.keras.models import load_model
from app.plugin import BasePlugin, PluginInfo
from .badge import BADGE

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ActiveLearningPlugin(BasePlugin):
    def __init__(self):
        self._info = PluginInfo(
            name="Active Learning",
            version="0.1",
            description="Selects samples from pool using BADGE based on a pretrained model.",
            arguments={
                "experiment_name": {
                    "name": "experiment_name",
                    "type": "str",
                    "description": "Name of the MLFlow experiment to retrieve the model from.",
                    "optional": False,
                },
                "query_size": {
                    "name": "query_size",
                    "type": "int",
                    "description": "Number of samples to select from the pool.",
                    "optional": True
                }
            },
            datasets={
                "pool": {
                    "name": "pool",
                    "description": "Unlabeled pool data to predict on."
                }
            },
            model_return_type=None
        )
        self.config: Dict = {}

    def init(self) -> PluginInfo:
        print("Initializing ActiveLearningPlugin")
        return self._info

    def run(self, *args, **kwargs) -> Optional[List[int]]:
        experiment_name = kwargs["experiment_name"]
        temp_dir = kwargs["temp_dir"]
        query_size = int(kwargs["query_size"]) if kwargs.get("query_size") is not None else 100

        print(f"Running plugin with arguments: {kwargs}")
        dataset_name = kwargs["datasets"]["pool"]

        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        print(f"Loading config from {config_path}")
        with open(config_path) as f:
            self.config = json.load(f)

        print("Retrieving model artifact")
        model_dir = self._connector.get_artifact(
            os.path.join(temp_dir, "emotion_model"),
            model_name=experiment_name,
            experiment_name=experiment_name,
            artifact_path="keras"
        )

        keras_path = glob.glob(os.path.join(model_dir, "*.keras"))
        if not keras_path:
            raise RuntimeError("Geen .keras model gevonden in de map.")
        print(f"Gekozen modelpad: {keras_path[0]}")
        model = load_model(keras_path[0])

        pool_dir = os.path.join(temp_dir, "pool")
        print(f"Loading pool dataset '{dataset_name}' into {pool_dir}")
        self._datastore.load_folder_to_filesystem(dataset_name, pool_dir)

        print("Loading images...")
        X_pool = self.load_images_from_folder(pool_dir)
        print(f"{X_pool.shape[0]} afbeeldingen geladen.")

        X_pool = X_pool / 255.0
        print("Afbeeldingen genormaliseerd")

        print("Initialiseren van BADGE...")
        badge = BADGE(model=model, pool_dataset=X_pool, batch_size=32, num_samples=query_size)
        selected_indices = badge.select()
        selected_indices = [int(i) for i in selected_indices]
        print(f"Geselecteerde indices (converted to int): {selected_indices}")

        # Opslaan als selection.json
        filenames = sorted([
            f for f in os.listdir(pool_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        selection_data = {
            "selected_indices": selected_indices,
            "selected_filenames": [filenames[i] for i in selected_indices],
            "timestamp": datetime.datetime.now().isoformat(),
            "total": len(X_pool)
        }

        selection_dir = os.path.join(temp_dir, "selection")
        os.makedirs(selection_dir, exist_ok=True)

        selection_path = os.path.join(selection_dir, "selection.json")
        with open(selection_path, "w") as f:
            json.dump(selection_data, f)

        self._datastore.upload_folder_to_datastore(
            folder_path=selection_dir,
            dataset_name=dataset_name,
            object_name="selection"
        )

        return selected_indices

    def load_images_from_folder(self, folder_path):
        import cv2
        image_list = []
        image_height = self.config["image_height"]
        image_width = self.config["image_width"]

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (image_width, image_height))
                    image_list.append(img)

        return np.array(image_list)
