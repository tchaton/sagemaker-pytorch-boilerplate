# The name of our algorithm
image=$1
MODEL=$2
DATASET=$3

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

cd container
cp -r ../deployement .
cp -r ../train deployement
cp -r ../src deployement/
pip install poetry
poetry export -f requirements.txt -o requirements.txt --without-hashes 
python render_docker.py
chmod +x deployement/train
chmod +x deployement/serve

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
# specifically setting to us-east-2 since during the pre-release period, we support only that region.
region=${region:-eu-west-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build --build-arg MODEL=$MODEL --build-arg DATASET=$DATASET -t ${image} .
docker tag ${image}:latest ${fullname}

aws ecr get-login-password \
    --region ${region} \
| docker login \
    --username AWS \
    --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

docker push ${fullname}
# Cleaning
rm -r deployement
rm Dockerfile
rm requirements.txt
echo ${fullname}
