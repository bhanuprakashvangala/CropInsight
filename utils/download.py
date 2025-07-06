import os
import urllib.request
import zipfile


def download_file(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {url} to {destination}...")
        urllib.request.urlretrieve(url, destination)
        print("Download finished.")
    else:
        print(f"{destination} already exists.")


def download_and_extract(url, extract_to, zip_path="temp.zip", force=False):
    if force or not os.path.exists(extract_to) or not os.listdir(extract_to):
        print(f"Downloading and extracting {url}...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
        print("Extraction finished.")
    else:
        print(f"{extract_to} already exists.")
