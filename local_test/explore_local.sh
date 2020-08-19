#!/bin/sh

image=$1

docker run -it -v $(pwd)/test_dir:/opt/ml -p 8080:8080 --rm ${image} bash
