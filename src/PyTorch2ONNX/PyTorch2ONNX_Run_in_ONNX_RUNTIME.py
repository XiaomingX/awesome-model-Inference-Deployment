import argparse
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import onnx
import onnxruntime

from utils import check_dir, torchtensor2numpy


class SuperResolutionNet(nn.Module):
    """ Super Resolution model definition in PyTorch."""
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


def export_model_to_onnx(model, dummy_input, onnx_save_path, check_onnx_model=True):
    """Export the PyTorch model to ONNX format."""
    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    if check_onnx_model:
        onnx_model = onnx.load(onnx_save_path)
        onnx.checker.check_model(onnx_model)


def verify_onnx_runtime(onnx_path, dummy_input, torch_output):
    """Verify ONNX Runtime output matches PyTorch output."""
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: torchtensor2numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(torchtensor2numpy(torch_output), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def run_model_in_onnx_runtime(onnx_path, img_path, img_save_path):
    """Run ONNX model on an image using ONNX Runtime."""
    img = Image.open(img_path)
    img = transforms.Resize([224, 224])(img)
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()
    img_y = transforms.ToTensor()(img_y).unsqueeze(0)

    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: torchtensor2numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = Image.fromarray(np.uint8((ort_outs[0][0] * 255.0).clip(0, 255)[0]), mode='L')

    final_img = Image.merge("YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
    final_img.save(img_save_path)


def main(args):
    # Load and prepare the super-resolution model
    model = SuperResolutionNet(upscale_factor=3)
    if torch.cuda.is_available():
        map_location = None
    else:
        map_location = lambda storage, loc: storage
    model.load_state_dict(model_zoo.load_url(
        'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth',
        map_location=map_location
    ))
    model.eval()

    # Prepare input to the model
    dummy_input = torch.randn(1, 1, 224, 224, requires_grad=True)
    torch_output = model(dummy_input)

    # Export model to ONNX
    export_model_to_onnx(model, dummy_input, args.onnx_save_path, args.check_onnx_model)

    # Verify ONNX Runtime results
    verify_onnx_runtime(args.onnx_save_path, dummy_input, torch_output)

    # Run model on an image
    run_model_in_onnx_runtime(args.onnx_save_path, args.img_path, args.img_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export a PyTorch model to ONNX and run it using ONNX Runtime.')
    parser.add_argument('--img_path', type=str, default='data/cat.jpg')
    parser.add_argument('--check_onnx_model', type=bool, default=True)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    check_dir(args.output_dir)
    args.onnx_save_path = os.path.join(args.output_dir, 'super_resolution.onnx')
    args.img_save_path = os.path.join(args.output_dir, 'cat_superres_with_ort.jpg')

    main(args)
