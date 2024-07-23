"""This module is used to initialize/verify/download the example_data.

If the three example files are already in the subdirectory, their sha256 hashes are compared with the hashes stored
in the file_dict dictionary in this file (init.py).

If the example files are missing or corrupt, they will be downloaded from zenodo.org.
By default, the example files are not provided on GitHub or pypi.org to keep the program lightweight and to save space.

Only when you run the example files in the "example" module, these files are downloaded (3 x 50 MB).
After that, these files are also verified against the fixed hash in the file_dict dictionary.
"""
import hashlib
import requests
from importlib import resources

# (importlib.resources can only access modules, not simple folders,
# that is the reason why example_data contains an empty init.py

ZENODO_RECORD = "12795959"
URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
file_dict = {"light.h5": {"sha256": "21fee3f3efeb280ab4dcfcf310f3ae683c390b14064da9630ca596da03170557"},
             "dark1.h5": {"sha256": "578659e74deaee439d697e3aa9d009447b13126884a4be97464614c787aa6050"},
             "dark2.h5": {"sha256": "d58d0f18ceb9b3770059b3ba2dc02dbfbea3cc52806d4634a747a55d4d1f1f32"}, }


def download_example_data(key, my_file):
    print(f"Downloading file {key}...")
    resp = requests.get(URL + f"/{key}/content")
    with open(my_file, "wb") as f:
        f.write(resp.content)  # writing content to file


def check_example_date(key):
    # TODO: Check if this works
    my_file = resources.files("parrot.example.example_data").joinpath(key)
    if my_file.is_file():
        with my_file.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256")
        if file_dict[key]["sha256"] == digest.hexdigest():
            print(f"Found correct file: {key}.")
        else:
            print(f"File {key} seems to be corrupt.")
            download_example_data(key, my_file)
            with my_file.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256")
            if file_dict[key]["sha256"] == digest.hexdigest():
                print(f"Downloaded correct file: {key}")
            else:
                print("Even after downloading could not retrieve correct file!")
                assert FileNotFoundError()
    else:
        print(f"Could not find file {key}.")
        download_example_data(key, my_file)
        with my_file.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256")
        if file_dict[key]["sha256"] == digest.hexdigest():
            print(f"Downloaded correct file: {key}")
        else:
            print("Even after downloading could not retrieve correct file!")
            assert FileNotFoundError()


def initialize():
    for key in file_dict.keys():
        print("###")
        check_example_date(key)
