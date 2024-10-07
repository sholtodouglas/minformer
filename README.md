# minformer

Minimal transformer for arbtirary data (i.e. bio stuff!)

Heavily inspired / stuff lifted from the classic - MinGPT.

## Setup (local):

Get gcloud CLI https://cloud.google.com/sdk/docs/install-sdk

## Setup (GCP)

```
# The first time you run tpu-vm ssh below, it will create a cloud specific ssh key.
# Print the public key and save it into ssh keys in your GCP project.
cat /Users/sholto/.ssh/google_compute_engine.pub
```

## Creating and connecting to a TPU

```
export TPU_ZONE=us-central1-a
export TPU_SIZE=v3-8
export PROJECT_ID=learning-from-play-303306
export TPU_NAME=devbox1

gcloud compute tpus tpu-vm create $TPU_NAME --zone $TPU_ZONE --accelerator-type=$TPU_SIZE --version=tpu-ubuntu2204-base --project=$PROJECT_ID


gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $TPU_ZONE --project $PROJECT_ID -- -L 8888:localhost:8888
```

The best way to connect is with vscode's remote extension.

Make sure your ssh config that you use in vs code points to your gcp ssh key.

```
Host xxx.xxx.xxx.xx
    HostName xxx.xxx.xxx.xx

Host *
    IdentityFile /Users/sholto/.ssh/google_compute_engine
```

You can then use the remote editor to edit code, and push/pull to sync.

curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="~/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
poetry install --extras "tpu"

## Profiling

%pip install tensorboard-plugin-profile

```
from jax.profiler import trace
with trace("/tmp/profile"):
    loss, weights, opt_state, internals = step(weights, batch['x'], batch['segment_ids'], batch['y'], opt_state, 0)
    jax.block_until_ready(loss)
```

```
python3 -m tensorboard.main --logdir=/tmp/profile
```

## Run training

```
python3 projects/charformer/train.py --checkpoint_dir=/tmp/charformer_checkpoints/test_run --checkpoint_interval=1000

python3 projects/charformer/train.py --checkpoint_dir=/tmp/charformer_checkpoints/test_run --checkpoint_interval=1000 --resume_from_checkpoint
python3 -m tensorboard.main --logdir=/tmp/logs
```

## Improvements

- [ ] 2nd order optimizer. Can try Shampoo based on recent [AlgoPerf Benchmark Competition](https://mlcommons.org/2024/08/mlc-algoperf-benchmark-competition/)
- [ ] Add jax.checkpoint to reduce memory usage.
- [ ] Try bidirectional attention e.g. splashattention
- [ ] Implement some kind of causal convolution for long context lengths
- [ ] Optimization: fwd/backwards pass using bfloat16 but keep accumulation as f32
- [ ] Weight tying between embedding and output layer
- [ ] GeLU or SwiGLU
- [ ] LayerNorm, RMSNorm
- [ ] FlashAttention
- [ ] FlashAttention2
- [ ] Multi-query attention
- [ ] Grouped attention
- [ ] Look at any recent lama 3.1 tricks https://github.com/meta-llama/llama3/blob/main/llama/model.py
- [ ] Reflection fine tuning
- [ ] @jit forward()
