import numpy as np
import PIL
import torch
from torchvision import transforms
from torch.autograd import Variable
from .models import EncoderCell, Binarizer, DecoderCell

enc = EncoderCell().to("cpu")
enc.load_state_dict(torch.load('project/encoder_epoch_00000007.pth', map_location=torch.device('cpu')), strict=True)
bin = Binarizer().to("cpu")
bin.load_state_dict(torch.load('project/binarizer_epoch_00000007.pth', map_location=torch.device('cpu')), strict=True)
dec = DecoderCell().to("cpu")
dec.load_state_dict(torch.load('project/decoder_epoch_00000007.pth', map_location=torch.device('cpu')), strict=True)



def encode(img, bottleneck):
    codes = []
    img = transforms.ToTensor()(img).unsqueeze(0)
    height, width = 256, 256
    res = img-0.5
    e1 = (Variable(torch.zeros(1, 256, height // 4, width // 4)),
                   Variable(torch.zeros(1, 256, height // 4, width // 4)))
    e2 = (Variable(torch.zeros(1, 512, height // 8, width // 8)),
                   Variable(torch.zeros(1, 512, height // 8, width // 8)))
    e3 = (Variable(torch.zeros(1, 512, height // 16, width // 16)),
                   Variable(torch.zeros(1, 512, height // 16, width // 16)))
    d1 = (Variable(torch.zeros(1, 512, height // 16, width // 16)),
                   Variable(torch.zeros(1, 512, height // 16, width // 16)))
    d2 = (Variable(torch.zeros(1, 512, height // 8, width // 8)),
                   Variable(torch.zeros(1, 512, height // 8, width // 8)))
    d3 = (Variable(torch.zeros(1, 256, height // 4, width // 4)),
                   Variable(torch.zeros(1, 256, height // 4, width // 4)))
    d4 = (Variable(torch.zeros(1, 128, height // 2, width // 2)),
                   Variable(torch.zeros(1, 128, height // 2, width // 2)))
    iters = 0
    while iters == 0 or np.packbits(((np.stack(codes).astype(np.int8) + 1) // 2).reshape(-1)).nbytes < bottleneck:
        x, e1, e2, e3 = enc(res, e1, e2, e3)
        y = bin(x)
        out, d1, d2, d3, d4 = dec(y, d1, d2, d3, d4)
        res = res-out
        codes.append(y.data.cpu().numpy())
        print('Iter: {:02d}; Loss: {:.06f}'.format(iters, res.data.abs().mean()))
        iters += 1
    return np.packbits(((np.stack(codes).astype(np.int8) + 1) // 2).reshape(-1))

def decode(x, bottleneck):
    height, width = 256, 256
    d1 = (Variable(torch.zeros(1, 512, height // 16, width // 16)),
                   Variable(torch.zeros(1, 512, height // 16, width // 16)))
    d2 = (Variable(torch.zeros(1, 512, height // 8, width // 8)),
                   Variable(torch.zeros(1, 512, height // 8, width // 8)))
    d3 = (Variable(torch.zeros(1, 256, height // 4, width // 4)),
                   Variable(torch.zeros(1, 256, height // 4, width // 4)))
    d4 = (Variable(torch.zeros(1, 128, height // 2, width // 2)),
                   Variable(torch.zeros(1, 128, height // 2, width // 2)))
    codes = np.unpackbits(x)
    dim = codes.size//(32*16*16)
    codes = np.reshape(codes, [dim,32,16,16]).astype(np.float32) * 2 - 1
    codes = torch.from_numpy(codes)
    img = torch.zeros(1, 3, height, width) + 0.5
    for iters in range(dim):
        out, d1, d2, d3, d4 = dec(codes[iters:iters+1], d1, d2, d3, d4)
        img = img + out.data.cpu()
    return transforms.ToPILImage()(img.clamp(min=0, max=1).select(0,0))
