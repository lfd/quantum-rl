#  Docker instructions

## Build docker image

`docker build -t quantum-rl .`

## Run docker container

After building you can create a container with:

`docker run --name qrl -it --runtime=nvidia -v $PWD:/app quantum-rl`

The container stops, when exiting.

## Access container

To restart the stopped container call:

`docker start -i qrl`