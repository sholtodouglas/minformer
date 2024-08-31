# minformer
Minimal transformer for arbtirary data (i.e. bio stuff!)


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

gcloud compute tpus tpu-vm create $TPU_NAME --zone $TPU_ZONE --accelerator-type=$TPU_SIZE --version=v2-alpha --project=$PROJECT_ID


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
