# Whisper Model Deployment Instructions

## Introduction

Whisper is an automatic speech recognition (ASR) and speech translation model developed by OpenAI, trained on 680,000 hours of labeled audio using large-scale weak supervision. It is a Transformer-based encoder–decoder model that generalizes well across languages, accents, and domains without fine-tuning. Whisper supports both speech-to-text and speech-to-translation tasks and is available in multiple model sizes, including English-only and multilingual versions.

## Model

**Original model link**
- Tiny: https://huggingface.co/openai/whisper-tiny
- Tiny-en: https://huggingface.co/openai/whisper-tiny.en
- Base: https://huggingface.co/openai/whisper-base
- Base-en: https://huggingface.co/openai/whisper-base.en
- Small: https://huggingface.co/openai/whisper-small
- Small-en: https://huggingface.co/openai/whisper-small.en

### Instructions on how to export the model

This [whisper_export.py](./model/whisper_export.py) exports the encoder and decoder of OpenAI Whisper models to ONNX format.

**Main Features**
- Export the **encoder** and **decoder** of Whisper models.
- Supports dynamic input lengths to adapt to different inference requirements.

**How to Use**
1. Install [required dependencies](./requirements.txt) by running: "pip install -r requirements.txt".
2. Modify the model path to select different models.
3. Run the script to export models to the `onnx_models` directory.

**Output Files**
- `whisper_tiny_en_encoder.onnx` - Encoder
- `whisper_tiny_en_decoder.onnx` - Decoder

Adjust the model path and output directory as needed.

**Note:** Ensure to replace the line:
```python
encoder_hidden_states = torch.tensor(np.random.rand(1, 384, 384), dtype=torch.float32)  # tiny/tiny-en:384, base/base-en:512, small/small-en:768
```
with the appropriate dimensions based on the model:
- Tiny/Tiny-en: `384`
- Base/Base-en: `512`
- Small/Small-en: `768`

## Model Conversion

Set `export ADLA_LLM_HYBRID_INT8_BATCH_MATMUL=True` to enable int8 quantization for BatchMatMul; however, this may lead to reduced accuracy.

**Encoder Conversion**

[Conversion shell script](./model/convert_encoder.sh)

**Decoder Conversion**

modify `1,48` in `--input-shapes` for diffenent length

[Conversion shell script](./model/convert_decoder.sh)

