import torch
from collections import OrderedDict

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def load_generator(model_path="generator_epoch_90.pth", use_cuda=False):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    netG.load_state_dict(new_state_dict, strict=False)
    netG.eval()
    
    return netG, device
