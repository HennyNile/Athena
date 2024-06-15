import os

def get_filename(path: str):
    return os.path.split(path)[-1]

def get_stem(path: str):
    return os.path.splitext(get_filename(path))[0]