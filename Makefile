help:
	@cat Makefile

IMAGE_NAME=msr
WORKDIR=msr
GPU?=0
DOCKER_FILE=Dockerfile

build:
	docker build . -t $(IMAGE_NAME) --build-arg DIR=$(WORKDIR) -f $(DOCKER_FILE)

bash:
	xhost +local:
	docker run -e DISPLAY --gpus $(GPU) -it --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --rm -v "`pwd`:/$(WORKDIR)" $(IMAGE_NAME) bash
	xhost -local:
