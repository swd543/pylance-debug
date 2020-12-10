#!/usr/bin/env python3

# %%
import SimpleITK as sitk
from torch.nn.modules import module
from utils.dataset import *
from utils import visualizations
from torchvision import transforms

transform = transforms.Compose([
    MinMaxNormalize(),
    SitkToNumpy(),
    transforms.ToTensor()
])
dataset=Brain2019(transform=transform, label_transform=OAR_bounding_box_one())
# img, segs = dataset[0]
# visualizations.display_animation(np.moveaxis(segs['Chiasm'].cpu().numpy(),-1,0))

# %%
import torch
from model.locnet import LocNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import AverageMeter
import wandb
import tempfile
from tqdm import tqdm
# print(img.unfold(1,512,512).shape)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

valid_dataset=Brain2019(transform=transform, label_transform=OAR_bounding_box_one(), train=False, valid_labels=['Chiasm'])

wandb.init(project='test', dir=tempfile.gettempdir())
print(f'Using cuda {torch.version.cuda}, cudnn version {torch.backends.cudnn.version()}, pytorch version {torch.__version__}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LocNet().to(device)
model = nn.DataParallel(model)
loss = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
history = AverageMeter('l1loss')

_metrics_to_collect = {'loss':loss}
_train_metrics={k:AverageMeter(f'train_{k}') for k in _metrics_to_collect.keys()}
_valid_metrics={k:AverageMeter(f'train_{k}') for k in _metrics_to_collect.keys()}

kernel_size = (1,512,512)

epochs=10
for e in range(epochs+1):
    wandb.log({'epoch':e, 'lr':[p['lr'] for p in optimizer.param_groups][-1]})

    model.train()
    with tqdm(dataset, desc=f'[{e}/{epochs}] Train', ascii=True) as progress, torch.enable_grad():
        for img, segs in progress:
            img = img.to(device)
            img = img.unfold(*kernel_size)
            segs = {k:v.to(device).unfold(*kernel_size) for k,v in segs.items() if k=='Chiasm'}
            optimizer.zero_grad()
            predictions = model(img)
            losses = loss(predictions, segs['Chiasm'])
            losses.backward()
            optimizer.step()

            for i,(k,v) in enumerate(_metrics_to_collect.items()):
                _train_metrics[k].update(losses.item())

            log_dict={v.name:v.avg for v in _train_metrics.values()}
            wandb.log(log_dict)
            progress.set_postfix(log_dict)
        for m in _train_metrics.values():
            m.commit()

    model.eval()
    with tqdm(valid_dataset, desc=f'[{e}/{epochs}] Valid', ascii=True) as progress, torch.no_grad():
        for img, segs in progress:
            img = img.to(device)
            img = img.unfold(*kernel_size)
            segs = {k:v.to(device).unfold(*kernel_size) for k,v in segs.items() if k=='Chiasm'}
            optimizer.zero_grad()
            predictions = model(img)
            losses = loss(predictions, segs['Chiasm'])
            losses.backward()
            optimizer.step()

            for i,(k,v) in enumerate(_metrics_to_collect.items()):
                _valid_metrics[k].update(losses.item())

            log_dict={v.name:v.avg for v in _valid_metrics.values()}
            wandb.log(log_dict)
            progress.set_postfix(log_dict)
        for m in _valid_metrics.values():
            m.commit()

class Buga():
    def __init__(self) -> None:
        x=['a']
        setattr(self, x)
h=Buga()