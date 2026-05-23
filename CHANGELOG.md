# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- commitizen will auto-update below this line -->


- infer source code media type and update deprecation handling in STAC items
- add ZenML integration for model promotion and STAC catalog synchronization
- implement STAC catalog management and builders for datasets and models

## v0.3.0 (2026-05-18)

### Feat

- update zenml version to 0.94.2 in helmfile and values
- update artifact naming and versioning across multiple files
- **data**: add STAC item for buildings instance segmentation in Banepa
- **ci**: add model resolution script for CI workflows
- **probe**: add test probe on the models
- add build and deployment scripts for CLI image and local stack setup

### Fix

- **ci**: add upload prediction
- **ci**: predictions
- **workflow**: add PR input to concurrency group in test-model workflow
- **tests**: improve mock handling in TestResolveDirectory for s3 paths
- **ci**: k8s
- **ci**: test cases
- **ci**: test cases add url path for the test cases
- **ci**: k8s stack registration
- **stac**: fix the stac item on the folders name
- **labels**: fix the classification labels
- **data**: fix stac datasets move the preprocessing to the pipeline
- **installation**: get rid of kind and add simpler docker compose
- **hotfix**: patch the artifacts materialization function
- **hotfix**: add s3 acl extra args
- **patch**: fix disable the patch report for codecov

### Refactor

- **mirror**: add single mirror function into data that is replicated everywhere

### Perf

- **chore**: added single runner for all the models with their support

## v0.2.0 (2026-05-05)

### Feat

- **patch**: add option to only patch few items in the stac items

### Fix

- **label**: fix labels href
- **local-stac**: add local stac validation
- **knative**: fix knative being invoked where it is not installed
- **dev**: ci k8s
- **infra**: env variable conflict

## v0.1.1 (2026-05-03)

### Fix

- **models**: added models in fair packaging

## v0.1.0 (2026-05-03)

### Feat

- **api**: add stac api backend in the fair-models
- **predictor**: adds geomltoolkits in the predict api
- **knative**: model serve
- **loss**: adds loss history record
- **onnx**: adds onnx feature building
- **opentofu**: adds opentofu with dok8s implementation
- **test**: adds test cases for the base models and the validators
- **freeze**: encoder fixes and stac item
- **model-validation**: adds validation for weight
- **schema**: introduces stac schema in hosted env
- **dtataset**: stac item
- **api**: exoose high level api

### Fix

- **local**: stac api url edge case
- **validation**: fix validation on the hyperparam key spec
- **ci**: bump docker build version v7
- **inference**: add s3fs in inference dep list
- **distro**: fix inference image distroless
- **ci**: checks eerymodel
- **test**: drop test coverage for models
- **ci**: catalog
- **onnx**: checkpoint addition
- **ci**: fix the coverage report to be model specific
- **ci**: added codecovtoken
- **ci**: add kubernetics to add test
- **knative**: fixes public domain dns mapping
- **dockerfile**: add fix for setuptools
- **coverage**: 6o percent drop
- **docker**: fixes the setup builds with hatch vcs
- **coverage**: fix the pipeline and add the coverage ( ci integration pipeline )
- **hyperparam**: enforce required hyperparam for training and inference with epoch
- **bump**: latest version of model images
- **test**: added test cases providers
- **provider**: added providers option in the stac metadata for models too
- **patch**: patch the log metdata in test
- **test-cases**: failing on local mode hence check it with upath
- **env**: migrates .env to tf vars opentofu
- **basemodels**: add deprecated lifecycle
- **pin**: pin zenml version in the lock file
- **docker**: fixes ci tests
- **deps**: remove rasterio deps from the main lib
- **ci**: fixes test cases added each test cases
- **session**: fix session never waited problem
- **pytest**: fix the path listing discovery
- **client**: fixes bug on inference when it loads last artifacts
- **ci**: remove unnecessary zenml pipeline
- **hotfix**: weight resolve on string
- **zenml**: config
- **ci**: add correct path for the workflow ci
- **schemas**: ci not being served static json
- **i**: ci
- **dataset**: adds seg version
- **href**: fix the absolute uri being returned in stac items
- **db**: fix db creds parse
- **env**: fixes db multiple env issue
- **classification**: add pyproj to classification
- **yolo**: checkpoints issue in k8s pod

### Perf

- **chore-bump**: version of stac item

## v0.0.7 (2026-04-05)

### Feat

- **val**: add train val split info in stac
- **stac**: hyperparam with classfiication object detection
- **examples**: adds classification segmentation and detection example
- **justfile**: adds justfile instead of makefile

### Fix

- **ci-test**: add htttpx
- **k8s**: stac
- **ci**: k8s
- **docs**: fixes doc on model with split as model requirements
- **onnx**: add onnx version pin
- **ci**: fixes ci chceks also includes the all run exampoles
- **onnx**: fixes inference on ci with dockerfile
- **fix**: ci prediction
- **ci-k8s**: just
- **ci**: abs path
- **cii**: relative path
- **ci**: makefile command
- **version**: fixes latest version
- **version**: added versioning self lib
- **temp**: temp fix for docker image
- **docker**: fixes docker version
- **sample**: fixes docker index url and sample size

## v0.0.6 (2026-04-01)

### Feat

- **stac**: cards
- **dataset**: versioning for dataset as well
- **dok8s**: adds k8s stac for digital ocean

### Fix

- **python**: fixes python version bump
- **docker**: image size trim only cpu build
- **metric**: fixes bug on metric evaluation moves to the fairopspy
- **auth**: ci k8s zenml
- **ci**: login in zenml k8s ci server disable
- **ci**: k8s kind config for taining
- **k8s**: adds fix for k8s deps in dockerfile

## v0.0.5 (2026-03-02)

### Fix

- **dockerfile**: fix double line in dockerfile for example
- **pyops**: upgarde docker to fair pyops
- **labels**: fix label of infra name in kind cluster

## v0.0.4 (2026-03-02)

### Feat

- **timeout**: adds timeout in ci for docker builds
- **docs**: adds mkdocs setup
- **validate**: model validation with pipeline args
- **mlflow**: adds mflow in pipeline

### Fix

- **stac**: fixes image of model remote uri
- **readme**: fixes readme validation on model update , fixes k8s source allocation to make sure workers are free !
- **license**: fixes license to restrict agpl
- **validation**: geom validation on dataset and model
- **zenml**: adds zenml client side patch
- **validation**: geom validation for model
- **docker**: adds underlying libs temp resolution
- **label**: fixes label name for the nodes
- **helm**: adds hemlfile config
- **gpu**: k8s gpu support with nvkind
- **prot**: fixes port forward foreground msgs

## v0.0.3 (2026-02-25)

### Fix

- **precommit**: fixes ruff version

## v0.0.2 (2026-02-25)

## v0.0.1 (2026-02-25)

### Fix

- **pkg-name**: fix package name being too similar

## v0.1.0 (2026-04-05)

### Feat

- **val**: add train val split info in stac
- **stac**: hyperparam with classfiication object detection
- **examples**: adds classification segmentation and detection example
- **justfile**: adds justfile instead of makefile
- **stac**: cards
- **dataset**: versioning for dataset as well
- **dok8s**: adds k8s stac for digital ocean

### Fix

- **ci-test**: add htttpx
- **k8s**: stac
- **ci**: k8s
- **docs**: fixes doc on model with split as model requirements
- **onnx**: add onnx version pin
- **ci**: fixes ci chceks also includes the all run exampoles
- **onnx**: fixes inference on ci with dockerfile
- **fix**: ci prediction
- **ci-k8s**: just
- **ci**: abs path
- **cii**: relative path
- **ci**: makefile command
- **version**: fixes latest version
- **version**: added versioning self lib
- **temp**: temp fix for docker image
- **docker**: fixes docker version
- **sample**: fixes docker index url and sample size
- **docker**: image size trim only cpu build
- **metric**: fixes bug on metric evaluation moves to the fairopspy

## v0.0.6 (2026-04-01)

### Fix

- **python**: fixes python version bump
- **auth**: ci k8s zenml
- **ci**: login in zenml k8s ci server disable
- **ci**: k8s kind config for taining
- **k8s**: adds fix for k8s deps in dockerfile

## v0.0.5 (2026-03-02)

### Fix

- **dockerfile**: fix double line in dockerfile for example
- **pyops**: upgarde docker to fair pyops
- **labels**: fix label of infra name in kind cluster

## v0.0.4 (2026-03-02)

### Feat

- **timeout**: adds timeout in ci for docker builds
- **docs**: adds mkdocs setup
- **validate**: model validation with pipeline args
- **mlflow**: adds mflow in pipeline

### Fix

- **stac**: fixes image of model remote uri
- **readme**: fixes readme validation on model update , fixes k8s source allocation to make sure workers are free !
- **license**: fixes license to restrict agpl
- **validation**: geom validation on dataset and model
- **zenml**: adds zenml client side patch
- **validation**: geom validation for model
- **docker**: adds underlying libs temp resolution
- **label**: fixes label name for the nodes
- **helm**: adds hemlfile config
- **gpu**: k8s gpu support with nvkind
- **prot**: fixes port forward foreground msgs

## v0.0.3 (2026-02-25)

### Fix

- **precommit**: fixes ruff version

## v0.0.2 (2026-02-25)

## v0.0.1 (2026-02-25)

### Feat

- infer source code media type and update deprecation handling in STAC items
- add ZenML integration for model promotion and STAC catalog synchronization
- implement STAC catalog management and builders for datasets and models

### Fix

- **pkg-name**: fix package name being too similar
