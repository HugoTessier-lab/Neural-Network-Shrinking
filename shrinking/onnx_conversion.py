import torch


def to_onnx(model, path, image_shape, device):
    model(torch.rand(image_shape).to(device))
    torch.onnx.export(model, torch.rand(image_shape).to(device), path, verbose=True, opset_version=13,
                      do_constant_folding=True, export_params=True)
