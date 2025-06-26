# Running evaluations

To run evaluations, do the following:

1. Start the NeMo Framework container. Remember to mount the checkpoint's path:

```
srun --container-image=nvcr.io/nvidian/nemo:25.07.rc2 --container-mounts /local/path/to/checkpoints:/checkpoints/ --pty /bin/bash
```

2. Install the required evaluation package. Each script contains a comment at the top indicating which package to use. The available packages are:

```
pip install nvidia-lm-eval==25.5
pip install nvidia-bigcode-eval==25.5
pip install nvidia-simple-evals==25.5
pip install nvidia-bfcl==25.5
pip install nvidia-eval-factory-garak==25.5
```

3. Deploy your model:

```
CHECKPOINT_PATH="/checkpoints/llama-3_2-1b-instruct_v2.0"  
python /opt/NeMo/scripts/deploy/nlp/deploy_in_fw_oai_server_eval.py --nemo_checkpoint ${CHECKPOINT_PATH} --max_input_len 8192 &
```

4. Check the top of the evaluation script for required environment variables. These variables must be exported before running the evaluation. 

5. Run an evalution of your choice. For example:

```
python mmlu.py
```
