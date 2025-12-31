export ADLA_ENABLE_LLM_HYBRID_QUANTIZE=True
export ADLA_SET_EXTREME_VALUE=16000

ADLA_TOOL_PATH=/xxxx/adla-toolkit-binary-3.3.9.4/bin/
adla_convert=${ADLA_TOOL_PATH}adla_convert

$adla_convert --model-type onnx \
        --model ./xxx/whisper_tiny_en_encoder.onnx \
        --inputs "input_features" \
        --input-shapes  "1,80,3000" \
        --shape-with-batch "True" \
        --dtypes "float32" \
        --quantize-dtype int16 --outdir ./whisper/tiny-en/encoder_int16 \
        --disable-per-channel False \
        --source-file ./xxx/whisper_English_audios_npy_it167.txt \
        --inference-output-type float32 \
        --target-platform PRODUCT_PID0XA005  # replace with your device PID
