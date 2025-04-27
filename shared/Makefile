devices ?= 0
force-build ?= false
use-gpu ?= false

image-name = tidl-alike

ifeq ($(force-build),true)
	method = run
else
	method = no-build-run
endif

ifeq ($(use-gpu),true)
	gpu-options = --gpus '"device=$(devices)"'
else
	gpu-options =
endif

build:
	DOCKER_BUILDKIT=1 docker build -t $(image-name)-image:1.0 .

no-build-run:
	docker run -it -d --shm-size=4096m \
	-e DISPLAY=$(DISPLAY) \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	--network host $(gpu-options) \
	--mount type=bind,source="$(shell pwd)"/assets,target=/home/workdir/assets \
	--mount type=bind,source="$(shell pwd)"/ALIKE,target=/home/workdir/ALIKE \
	--mount type=bind,source="$(shell pwd)"/scripts,target=/home/workdir/scripts \
	--mount type=bind,source="$(shell pwd)"/components,target=/home/workdir/components \
	--mount type=bind,source="$(shell pwd)"/installation,target=/home/workdir/installation \
	--mount type=bind,source=/dev,target=/host/dev \
	--name $(image-name)-container $(image-name)-image:1.0

run: build no-build-run

start:
	make $(method)

exec:
	docker exec -it $(image-name)-container /bin/bash

stop:
	docker stop $(image-name)-container
	docker rm $(image-name)-container
