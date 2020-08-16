FROM ubuntu:bionic

LABEL maintainer="thomas.chaton.ai@gmail.com"

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --fix-missing --no-install-recommends\
    ca-certificates nginx wget libffi-dev libssl-dev build-essential libopenblas-dev\
    python3-pip python3-dev python3-venv python3-setuptools\
    git iproute2 procps lsb-release \
    libsm6 libxext6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip \
    && python3 -m pip install poetry \
    && poetry config virtualenvs.create false --local \
    && pip3 install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install setuptools>=41.0.0 \
    && rm -rf /root/.cache

RUN pip install absl-py==0.9.0
RUN pip install cachetools==4.1.1
RUN pip install certifi==2020.6.20
RUN pip install cffi==1.14.2
RUN pip install chardet==3.0.4
RUN pip install click==7.1.2
RUN pip install enum-compat==0.0.3
RUN pip install flask==1.1.2
RUN pip install future==0.18.2
RUN pip install gevent==20.6.2
RUN pip install google-auth==1.20.1
RUN pip install google-auth-oauthlib==0.4.1
RUN pip install greenlet==0.4.16
RUN pip install grpcio==1.31.0
RUN pip install gunicorn==20.0.4
RUN pip install hydra-core==0.11.3
RUN pip install idna==2.10
RUN pip install importlib-metadata==1.7.0
RUN pip install itsdangerous==1.1.0
RUN pip install jinja2==2.11.2
RUN pip install joblib==0.16.0
RUN pip install markdown==3.2.2
RUN pip install markupsafe==1.1.1
RUN pip install numpy==1.19.1
RUN pip install oauthlib==3.1.0
RUN pip install omegaconf==1.4.1
RUN pip install packaging==20.4
RUN pip install pandas==1.1.0
RUN pip install pillow==7.2.0
RUN pip install protobuf==3.13.0
RUN pip install psutil==5.7.2
RUN pip install pyasn1==0.4.8
RUN pip install pyasn1-modules==0.2.8
RUN pip install pycparser==2.20
RUN pip install pyparsing==2.4.7
RUN pip install python-dateutil==2.8.1
RUN pip install pytorch-lightning==0.8.5
RUN pip install pytz==2020.1
RUN pip install pyyaml==5.3.1
RUN pip install requests==2.24.0
RUN pip install requests-oauthlib==1.3.0
RUN pip install rsa==4.6
RUN pip install scikit-learn==0.23.2
RUN pip install scipy==1.5.2
RUN pip install six==1.15.0
RUN pip install tensorboard==2.3.0
RUN pip install tensorboard-plugin-wit==1.7.0
RUN pip install threadpoolctl==2.1.0
RUN pip install torch==1.6.0
RUN pip install torch-model-archiver==0.2.0
RUN pip install torchserve==0.2.0
RUN pip install tqdm==4.48.2
RUN pip install urllib3==1.25.10
RUN pip install werkzeug==1.0.1
RUN pip install wheel==0.35.1
RUN pip install zipp==3.1.0
RUN pip install zope.event==4.4
RUN pip install zope.interface==5.1.0

RUN rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY src /opt/program
WORKDIR /opt/program
