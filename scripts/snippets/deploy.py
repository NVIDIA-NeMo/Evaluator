# File deploy.py

from nemo_evaluate.api import deploy

CHECKPOINT_PATH="/path/to/checkpoint/llama-3_2-1b-instruct_v2.0"

deploy(
    nemo_checkpoint=CHECKPOINT_PATH,
    max_input_len=8192,
)