import torch
from transformers import BertTokenizer
from transformers.onnx import export
from transformers.onnx import FeaturesManager
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def convert_to_onnx(model_path, onnx_path):
    """Convert PyTorch BERT model to ONNX format."""
    # Load the model
    from src.models.bert_model import BertModel
    model = BertModel()
    model.load_model(model_path)
    model.eval()

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Dummy inputs for export
    dummy_input = tokenizer("This is a test email", return_tensors="pt")
    dummy_features = torch.randn(1, 6)  # 6 features

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask'], dummy_features),
        onnx_path,
        input_names=['input_ids', 'attention_mask', 'features'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'features': {0: 'batch_size'}},
        opset_version=11
    )
    print(f"Model converted to ONNX and saved at {onnx_path}")

def quantize_onnx_model(onnx_path, quantized_path):
    """Quantize the ONNX model."""
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
    print(f"Quantized model saved at {quantized_path}")

if __name__ == '__main__':
    model_path = 'models/fine_tuned_bert_model'
    onnx_path = 'models/bert_model.onnx'
    quantized_path = 'models/bert_model_quantized.onnx'

    convert_to_onnx(model_path, onnx_path)
    quantize_onnx_model(onnx_path, quantized_path)
