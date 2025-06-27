# File deploy.py

from nemo_eval.api import deploy


CHECKPOINT_PATH = "/checkpoints/llama-3_2-1b-instruct_v2.0"

if __name__ == "__main__":
    deploy(
        nemo_checkpoint=CHECKPOINT_PATH,
        max_input_len=8192,
    )
