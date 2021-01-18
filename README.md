# SoftGym
<a href="https://sites.google.com/view/softgym/home">SoftGym</a> is a set of benchmark environments for deformable object manipulation including tasks involving fluid, cloth and rope. It is built on top of the Nvidia FleX simulator and has standard Gym API for interaction with RL agents. 

## Using Docker
The provided Dockerfile is based on the [pre-built image for softgym](https://hub.docker.com/layers/xingyu/softgym/latest/images/sha256-29a9f674cf3527e645a237facdfe4b5634c23cd0f1522290e0a523308435ccaa?context=explore) which in turn uses CUDA 9.2. This codebase is tested with Ubuntu 20.04 LTS and Nvidia driver version 450.102.04

## Prerequisites

- Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart) (Docker versions earlier than 19.03 require nvidia-docker2 and the `--runtime=nvidia` flag. On versions including and after 19.03, you will use the `nvidia-container-toolkit` package and the `--gpus all` flag)

## Building and running Dockerfile

1. run `make build` to build the docker image
2. run `make bash` to run the image with an interactive bash session (step 2 also includes step 1)
3. run `source setup.sh` to setup the `softgym conda environment` and compile the python bindings for `pyflex`
 
## SoftGym Environments
|Image|Name|Description|
|----------|:-------------|:-------------|
|![Gif](./examples/ClothDrop.gif)|[DropCloth](softgym/envs/cloth_drop.py) | Lay a piece of cloth in the air flat on the floor|
|![Gif](./examples/ClothFold.gif)|[FoldCloth](softgym/envs/cloth_fold.py) | Fold a piece of flattened cloth in half|
|![Gif](./examples/ClothFlatten.gif)|[SpreadCloth](softgym/envs/cloth_flatten.py)| Spread a crumpled cloth on the floor|
|![Gif](./examples/PourWater.gif)|[PourWater](softgym/envs/pour_water.py)| Pour a cup of water into a target cup |
|![Gif](./examples/PassWater.gif)|[TransportWater](softgym/envs/pass_water.py)| Move a cup of water to a target position as fast as possible without spilling out the water|
|![Gif](./examples/RopeFlatten.gif)|[StraightenRope](softgym/envs/rope_flatten.py)| Straighten a rope starting from a random configuration|
|![Gif](./examples/PourWaterAmount.gif)|[PourWaterAmount](softgym/envs/pour_water_amount.py)| This task is similar to PourWater but requires a specific amount of water poured into the target cup. The required water level is indicated by a red line.|
|![Gif](./examples/ClothFoldCrumpled.gif)|[FoldCrumpledCloth](softgym/envs/cloth_fold_crumpled.py)| This task is similar to FoldCloth but the cloth is initially crumpled| 
|![Gif](./examples/ClothFoldDrop.gif)|[DropFoldCloth](softgym/envs/cloth_fold_drop.py)| This task has the same initial state as DropCloth but requires the agent to fold the cloth instead of just laying it on the ground|
|![Gif](./examples/RopeConfiguration.gif)|[RopeConfiguration](softgym/envs/rope_configuration.py)| This task is similar to StraightenCloth but the agent needs to manipulate the rope into a specific configuration from different starting locations.|
   
To have a quick view of different tasks listed in the paper (with random actions), run the following commands:
For SoftGym-Medium:  
- TransportWater: `python examples/random_env.py --env_name PassWater`
- PourWater: `python examples/random_env.py --env_name PourWater`
- StraightenRope: `python examples/random_env.py --env_name RopeFlatten`
- SpreadCloth: `python examples/random_env.py --env_name ClothFlatten`
- FoldCloth: `python examples/random_env.py --env_name ClothFold`
- DropCloth: `python examples/random_env.py --env_name ClothDrop`  

For SoftGym-Hard:  
- PourWaterAmount: `python examples/random_env.py --env_name PourWaterAmount`
- FoldCrumpledCloth: `python examples/random_env.py --env_name ClothFoldCrumpled`
- DropFoldCloth: `python examples/random_env.py --env_name ClothFoldDrop`
- RopeConfiguration: `python examples/random_env.py --env_name RopeConfiguration`  

Please refer to `softgym/registered_env.py` for the default parameters and source code files for each of these environments.

## Cite
If you find this codebase useful in your research, please consider citing:
```
@inproceedings{corl2020softgym,
 title={SoftGym: Benchmarking Deep Reinforcement Learning for Deformable Object Manipulation},
 author={Lin, Xingyu and Wang, Yufei and Olkin, Jake and Held, David},
 booktitle={Conference on Robot Learning},
 year={2020}
}
```

## References
- NVIDIA FleX - 1.2.0: https://github.com/NVIDIAGameWorks/FleX
- Our python interface builds on top of PyFleX: https://github.com/YunzhuLi/PyFleX
