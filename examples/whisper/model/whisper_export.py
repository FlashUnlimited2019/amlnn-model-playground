# This script exports a Whisper model's encoder and decoder to ONNX format.

import torch
import torch.nn as nn
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

whisper_tiny_en = "openai/whisper-tiny.en"  # Replace with your local model path.

os.makedirs("./onnx_models/whisper-tiny-en_encoder/", exist_ok=True)
os.makedirs("./onnx_models/whisper-tiny-en_decoder/", exist_ok=True)
encoder_onnx_path = "./onnx_models/whisper-tiny-en_encoder/whisper_tiny_en_encoder.onnx"
decoder_onnx_path = "./onnx_models/whisper-tiny-en_decoder/whisper_tiny_en_decoder.onnx"

processor = WhisperProcessor.from_pretrained(whisper_tiny_en)
model = WhisperForConditionalGeneration.from_pretrained(whisper_tiny_en)


encoder = model.model.encoder

input_features = torch.tensor(np.random.rand(1, 80, 3000), dtype=torch.float32)

# Export encoder
torch.onnx.export(
    encoder,
    (input_features,),
    encoder_onnx_path,
    input_names=["input_features"],
    output_names=["encoder_output"],
    dynamic_axes={"input_features": {2: "input_length"}},
    opset_version=14,
    verbose=True,
)

class WhisperDecoderWrapper(nn.Module):
    def __init__(self, decoder, projection):
        super().__init__()
        self.decoder = decoder
        self.projection = projection

    def forward(self, input_ids, encoder_hidden_states):
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        return self.projection(decoder_outputs.last_hidden_state)


decoder_wrapper = WhisperDecoderWrapper(model.model.decoder, model.proj_out)

decoder_input_ids = torch.tensor(np.random.rand(1, 64), dtype=torch.int64)
encoder_hidden_states = torch.tensor(np.random.rand(1, 384, 384), dtype=torch.float32)  # tiny:384, base:512, small:768

# Export decoder
torch.onnx.export(
    decoder_wrapper,
    (decoder_input_ids, encoder_hidden_states),
    decoder_onnx_path,
    input_names=["input_ids", "encoder_hidden_states"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {1: "sequence_length"},
        "encoder_hidden_states": {1: "sequence_length"},
    },
    opset_version=14,
    verbose=True,
)
