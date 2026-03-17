import os
import requests  # Make sure requests is installed
import json
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm

def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    ## We have reached here until now ---> we have downloaded the files on our local machine.

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def download_file(url, destination):
    try:
        # Send a GET request to download the file, disabling SSL verification
        response = requests.get(url, stream=True, verify=False)

        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("content-length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        progress_bar_description = url.split("/")[-1]  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Iterate over the file data in chunks
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))  # Update progress bar
                    file.write(chunk)  # Write the chunk to the file

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):

    params = {
        "blocks": [{} for _ in range(settings["n_layer"])]
    }

    for name, _ in tf.train.list_variables(ckpt_path):

        array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Remove "model/" prefix
        name = name.replace("model/", "")
        parts = name.split("/")

        # 🧠 Case 1: Transformer block (h0, h1, ...)
        if parts[0].startswith("h"):
            layer = int(parts[0][1:])
            target = params["blocks"][layer]

            for key in parts[1:-1]:
                target = target.setdefault(key, {})

            target[parts[-1]] = array

        # 🧠 Case 2: Top-level weights (wte, wpe, ln_f)
        else:
            if len(parts) == 1:
                # Example: wte or wpe
                params[parts[0]] = array
            else:
                # Example: ln_f/b or ln_f/g
                target = params.setdefault(parts[0], {})
                for key in parts[1:-1]:
                    target = target.setdefault(key, {})

                target[parts[-1]] = array

    return params


