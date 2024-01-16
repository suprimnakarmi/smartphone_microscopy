# smartphone_microscopy
AI-Assisted Smartphone Microscopy for automated detection of Giardia and Cryptosporidium cysts.

This repository contains all the configurations and utilities used to train the models for AI Assisted Smartphone Microscopy. 

Please find the dataset [here](https://zenodo.org/record/7813183).


# mmdetection for Faster RCNN and RetinaNet
## Installing dependencies

1. Create a virtual environment.
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. Install `torch==1.13.0`.
    ```sh
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
    ```

3. Install `mmcv==1.7.1`.
    ```sh
    pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
    ```

4. This repo uses `mmdetection v2.27.0`. Clone the mmdetection repo at `v2.27.0` and install.
    ```sh
    git clone --depth 1 --branch v2.27.0 https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    pip install -v -e .
    ```
    
## Model Training and Prediction

The config file used for training and testing are inside `configs/faster_rcnn` and `configs/retinanet`.

### Training 

```shell
python mmdetection/tools/train.py <config_file>
```

### Prediction

```shell
python mmdetection/tools/test.py <config_file> \
        --gpu-id 0 \
        <checkpoint-path-for-fold> \
        --format-only \
        --options \
        "jsonfile_prefix=<output_folder>/<fold>/results_test"
```


The results are saved inside `<output_folder>/<fold>/` folder named `results_test.bbox.json`.
The code to analyze the results are present inside [`notebooks/result_analysis.ipynb`](notebooks/result_analysis.ipynb) notebook.

# Yolov8

## Installing dependencies

Download Yolov8s model weights from [here](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt).

```sh
pip install ultralytics
```

Change the fold number inside respective data and model files in [configs/yolov8/data](configs/yolov8/data) and [configs/yolov8/model](configs/yolov8/model) folders.

## Training and prediction

Run the respective file inside [configs/yolov8/model](configs/yolov8/model) folder.

```sh
python configs/yolov8/model/brightfield_reference.py
```

The training and validation info will be saved inside `runs` folder.

Refer to https://github.com/ultralytics/ultralytics to further costumize yolov8 runs.