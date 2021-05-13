#  Docker instructions

## Build docker image

`docker build -t quantum-rl:nvidia-tf .`

## Run docker container

After building you can create a container using the image with:

`docker run --name tfq -it --runtime=nvidia --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 quantum-rl:nvidia-tf`

After exiting the container stops, but still exists.

## Access container

To start the stopped container call:

`docker start -i tfq`