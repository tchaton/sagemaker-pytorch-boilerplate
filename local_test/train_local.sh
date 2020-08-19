#!/bin/sh

IMAGE=$1
ARGS=(${@: 2})
echo $ARGS

rm -r test_dir/model
rm -r test_dir/output

mkdir -p test_dir/model
mkdir -p test_dir/output

docker run -v $(pwd)/test_dir:/opt/ml --rm ${IMAGE} train ${ARGS}
