import torch
from transformers import CLIPModel
import argparse
import os

class TextModelWrapper(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.text_model = clip.text_model
        self.text_projection = clip.text_projection

    def forward(self, input_ids):
        outputs = self.text_model(
            input_ids=input_ids,
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        return text_features

class VisionModelWrapper(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection

    def forward(self, pixel_values):
        outputs = self.vision_model(
            pixel_values=pixel_values,
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        image_features = self.visual_projection(pooled_output)
        return image_features

def export_models(model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')
    clip = CLIPModel.from_pretrained(model_path).to(device)
    clip.eval()

    # Export Text Model
    text_wrapper = TextModelWrapper(clip).to(device)
    seq_len = clip.config.text_config.max_position_embeddings
    dummy_input_ids = torch.randint(0, clip.config.text_config.vocab_size, (1, seq_len), dtype=torch.long)

    torch.onnx.export(
        text_wrapper,
        (dummy_input_ids),
        os.path.join(output_dir, "text_model.onnx"),
        input_names=['input_ids'],
        output_names=['text_embeds'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'text_embeds': {0: 'batch_size'}
        },
        opset_version=15
    )

    # Export Vision Model
    vision_wrapper = VisionModelWrapper(clip).to(device)
    img_size = clip.config.vision_config.image_size
    dummy_pixel_values = torch.randn((1, 3, img_size, img_size), dtype=torch.float)

    torch.onnx.export(
        vision_wrapper,
        (dummy_pixel_values,),
        os.path.join(output_dir, "vision_model.onnx"),
        input_names=['pixel_values'],
        output_names=['image_embeds'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'},
            'image_embeds': {0: 'batch_size'}
        },
        opset_version=15
    )

    print("ONNX models exported successfully to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CLIP text and vision sub-models to ONNX")
    parser.add_argument(
        "--model_path",
        type=str,
        default="openai/clip-vit-base-patch32",  # Change to your CLIP model path
        help="Path to the pretrained CLIP model."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./onnx_export",
        help="Directory where the ONNX models will be saved."
    )
    args = parser.parse_args()
    export_models(args.model_path, args.output_dir)
