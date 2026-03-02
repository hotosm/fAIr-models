# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

<!-- commitizen will auto-update below this line -->


- infer source code media type and update deprecation handling in STAC items
- add ZenML integration for model promotion and STAC catalog synchronization
- implement STAC catalog management and builders for datasets and models

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
