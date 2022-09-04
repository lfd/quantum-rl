# Deep Reinforcement Learning with Variational Quantum Circuits

Code accompanying the paper: 

**Uncovering Instabilities in Variational-Quantum Deep Q-Networks**.
Maja Franz, Lucas Wolf, Maniraman Periyasamy, Christian Ufrecht, Daniel D. Scherer, Axel Plinge, Christopher Mutschler, Wolfgang Mauerer.
Journal of The Franklin Institute. Elsevier (Open Access). 2022.

## Training

To train an agent run: 
```
python3 train.py path.to.config
```
e.g.
```
python3 train.py tfq.reproduction.skolik.cartpole.skolik_hyper.baseline.gs.sc_enc
```

We provide several configuration files in `configs/`

A training process can be tracked with TensorBoard:

`tensorboard --logdir logs --host 0.0.0.0`

## Plotting

We provide all scripts and data, which were used to create the graphs in the paper.

Generate all figures:
```
cd plot/
./plot_all.sh 
```

Plot a single or multiple conducted experiments:
```
cd plot/
./plot_runs.sh
```

##  Docker

The Dockerimages contain all neccessary libraries and packages to run the training and plotting scripts. We provide one GPU and one CPU-only version.

### Build docker image

```
docker build -f Dockerfile_GPU -t quantum-rl-gpu .
```

or

```
docker build -f Dockerfile_CPU -t quantum-rl-cpu .
```

### Create docker container

```
docker run --name qrl-gpu -it -d --runtime=nvidia -v $PWD:/home/repro -p 6006:6006 quantum-rl-gpu
```

or

```
docker run --name qrl-cpu -it -d -v $PWD:/home/repro -p 6006:6006 quantum-rl-cpu
```

### Access container

```
docker exec -it qrl-cpu /bin/bash
```

or 

```
docker exec -it qrl-cpu /bin/bash
```

Restart a stopped container:

```
docker start qrl-gpu
```

or

```
docker start qrl-cpu
```
