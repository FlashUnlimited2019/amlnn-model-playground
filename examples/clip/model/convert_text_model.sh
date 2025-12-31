export ADLA_ENABLE_LLM_HYBRID_QUANTIZE=True
export ADLA_SET_RANDOM_MAX_VALUE=49408  # Use only with random data (set max input value); may cause major accuracy loss for other models (e.g., vision model).

ADLA_TOOL_PATH=/xxxx/adla-toolkit-binary-3.3.9.4/bin/
adla_convert=${ADLA_TOOL_PATH}adla_convert

$adla_convert --model-type onnx \
        --model ./xxx/text_model.onnx \
        --inputs "input_ids" \
        --input-shapes  "1,32" \
        --shape-with-batch "True" \
        --dtypes "int64" \
        --quantize-dtype int16 \
        --outdir ./clip-vit-base-patch32/2models/text_model/int16_model/ \
        --disable-per-channel False \
        --inference-input-type float32 \
        --inference-output-type float32 \
        --iterations 100 \
        --target-platform PRODUCT_PID0XA005  # replace with your device PID
        
        # Use this option to work with a custom dataset
        # --source-file xxxxx.txt \  