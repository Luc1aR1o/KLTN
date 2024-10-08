import glob
import os.path
import re
import sys

module_path = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))
SAMPLE_PATH = os.path.join(module_path, "samples")

def fetch_file(sample_path):
    with open(sample_path, "rb") as f:
        bytez = f.read()
    return bytez


def get_available_sha256():
    sha256list = []
    for fp in glob.glob(os.path.join(SAMPLE_PATH, "*")):
        fn = os.path.split(fp)[-1]
        # require filenames to be sha256
        result = re.match(r"^[0-9a-fA-F]{64}$", fn)
        if result:
            sha256list.append(result.group(0))
    # no files found in SAMLPE_PATH with sha256 names
    assert len(sha256list) > 0
    return sha256list
