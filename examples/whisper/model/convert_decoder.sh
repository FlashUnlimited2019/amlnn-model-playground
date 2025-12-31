export ADLA_ENABLE_LLM_HYBRID_QUANTIZE=True
export ADLA_SET_EXTREME_VALUE=16000

ADLA_TOOL_PATH=/xxxx/adla-toolkit-binary-3.3.9.4/bin/
adla_convert=${ADLA_TOOL_PATH}adla_convert

$adla_convert --model-type onnx \
        --model ./xxx/whisper_tiny_en_decoder.onnx \
        --inputs "input_ids encoder_hidden_states" \
        --input-shapes  "1,48#1,1500,384" \
        --shape-with-batch "True#True" \
        --dtypes "int64#float32" \
        --quantize-dtype int8 --outdir ./whisper/tiny-en/decoder_48_int8 \
        --disable-per-channel False \
        --source-file ./xxx/whisper-tiny-en_decoder_it310.txt \
        --inference-output-type float32 \
        --target-platform PRODUCT_PID0XA005  # replace with your device PID
