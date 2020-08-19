# sagemaker-pytorch-boilerplate

Production ML, as a field, has matured. It’s increasingly common for companies to have at least one model in production. As more teams deploy models, the conversation around tooling has shifted from “What gets the job done?” to “What does it take to deploy a model at production scale?”

This project is a boilerplate codebase to `train / serve / publish` Pytorch Model using AWS Sagemaker.

We aim at simplifying MLOps worflow by providing a template for production ready development, allowing ML engineer to focus uniquely on their models and datasets. 

We rely on [Hydra](https://hydra.cc) for elegantly configuring our application and [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), a lightweight PyTorch wrapper for ML researchers to scale their experiments with less boilerplate.

# How to use this project

This project implements a 1-layer MLP on iris dataset as a baby demo.

```bash
sh build_local_env.sh 3.7.8 # It will create a local env to ease local dev
```

```bash
sh build_and_push.sh {IMAGE_NAME} {MODEL} {DATASET}
# It will build the folder container and push the image to AWS Elastic Container Registry (ECR)
```

# Local development

## Training

Used to make quick dev.

```bash
source .venv/bin/activate
python src/train model={MODEL} dataset={DATASET}
```

or within docker image

Used to make sure the docker image is correcly working

```bash
sh local_test/train_local.sh ${IMAGE_NAME} ${ARGS_1} ${ARGS_2} ${ARGS_3} ...
```

## Local Serving

Terminal 1
```bash 
In:
sh build_and_push.sh {IMAGE_NAME} {MODEL} {DATASET}.
sh local_test/serve_local.sh {IMAGE_NAME}
```

``` bash
Out:
Starting the inference server with 4 workers.
[2020-08-19 11:41:31 +0000] [9] [INFO] Starting gunicorn 20.0.4
[2020-08-19 11:41:31 +0000] [9] [INFO] Listening at: unix:/tmp/gunicorn.sock (9)
[2020-08-19 11:41:31 +0000] [9] [INFO] Using worker: gevent
[2020-08-19 11:41:31 +0000] [13] [INFO] Booting worker with pid: 13
[2020-08-19 11:41:31 +0000] [14] [INFO] Booting worker with pid: 14
[2020-08-19 11:41:31 +0000] [15] [INFO] Booting worker with pid: 15
[2020-08-19 11:41:31 +0000] [16] [INFO] Booting worker with pid: 16
```

Terminal 2
```bash 
In:
sh local_test/predict.sh {SAMPLE_DATA} # Currently support only 'text/csv'
```


Train on AWS:
Run workflow.ipynb notebook

```
jupyter lab
```

# CAREFUL: Work in progress
