export ADLA_ENABLE_LLM_HYBRID_QUANTIZE=True

ADLA_TOOL_PATH=/xxxx/adla-toolkit-binary-3.3.9.4/bin/
adla_convert=${ADLA_TOOL_PATH}adla_convert

$adla_convert --model-type onnx \
        --model ./xxx/vision_model.onnx \
        --inputs "pixel_values" \
        --input-shapes  "1,3,224,224" \
        --shape-with-batch "True" \
        --dtypes "float32" \
        --quantize-dtype int8 \
        --outdir ./clip-vit-base-patch32/2models/vision_model_int8 \
        --disable-per-channel False \
        --source-file ./clip-vit-base-patch32_quant_datasets/2_models_vision_model_quant_datasets/it160data.txt \
        --inference-output-type float32 \
        --target-platform PRODUCT_PID0XA005  # replace with your device PID
