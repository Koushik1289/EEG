import os
import time
import urllib.request

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_file(url, dest):
    if not os.path.exists(dest):
        log(f"Downloading {dest}")
        urllib.request.urlretrieve(url, dest)
