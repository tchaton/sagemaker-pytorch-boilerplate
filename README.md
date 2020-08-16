# sagemaker-torch-template

Production ML, as a field, has matured. It’s increasingly common for companies to have at least one model in production. As more teams deploy models, the conversation around tooling has shifted from “What gets the job done?” to “What does it take to deploy a model at production scale?”

This template is a boiler template codebase to train / serve Pytorch Model using AWS Sagemaker.
ML Engineer will have to take care only about their model definition and dataset.

# How to use this project

```bash
sh build_local_env.sh 3.7.8 # It will create a local env to ease local dev
```

```bash
sh build_and_push.sh {IMAGE_NAME} # It will build the folder container and push the image to AWS Elastic Container Registry (ECR)
```

Train Locally:

Used to make quick dev.

```bash
source .venv/bin/activate
python container/src/train +model_type=model_template +model_name=my_template_model +dataset=dataset_template
```

or within docker image

Used to make sure the docker image is correcly working

```bash
cd container/local_test
train_local.sh ${IMAGE_NAME} ${ARGS_1} ${ARGS_2} ${ARGS_3} ...
```

Train on AWS:
Run workflow.ipynb notebook

```
jupyter lab
```

# CAREFUL: Work in progress
