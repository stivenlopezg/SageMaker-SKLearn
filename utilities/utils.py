import os


def download_artifact(s3_path: str, local_path: str):
    return os.system(command=f"aws s3 cp {s3_path} {local_path}")


def decompress_artifact(filepath: str):
    return os.system(command=f"tar xfv {filepath} -C model")
