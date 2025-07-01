# File deploy.py

import sys

from nemo_eval.api import deploy


if __name__ == "__main__":
    CHECKPOINT_PATH = sys.argv[1]
    deploy(
        nemo_checkpoint=CHECKPOINT_PATH,
        max_input_len=8192,
    )
