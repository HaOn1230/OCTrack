
This repository provides the official implementation of OCTrack.

Detailed instructions will be provided later.

## Installation

Create and activate a conda environment:

```bash
conda create -n OCTrack python=3.8
conda activate OCTrack
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Preparation

Put the tracking datasets in `./data`. It should look like:

```text
${PROJECT_ROOT}
|-- data
    |-- lasot
    |   |-- airplane
    |   |-- basketball
    |   |-- bear
    |   |-- ...
    |
    |-- got10k
    |   |-- test
    |   |-- train
    |   |-- val
    |
    |-- coco
    |   |-- annotations
    |   |-- images
    |
    |-- trackingnet
        |-- TRAIN_0
        |-- TRAIN_1
        |-- ...
        |-- TRAIN_11
        |-- TEST
```
