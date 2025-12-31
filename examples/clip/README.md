# OpenAI CLIP Model Deployment Instructions

## Introduction

CLIP is a model developed by OpenAI to study robustness in computer vision and enable zero-shot image classification. It learns to connect images and text, allowing generalization to unseen tasks without task-specific training.

## Model

**Original model link**

https://huggingface.co/openai/clip-vit-base-patch32

### Instructions on how to export the model  

- Export two ONNX models (separate vision model & text model) using [onnx_export_2models.py](./model/onnx_export_2models.py)

- Manual implementation of pre-processing and post-processing is required.

## Model Conversion

**vision model**

[Conversion shell script](./model/convert_vision_model.sh)

**text model**

[Conversion shell script](./model/convert_text_model.sh)
