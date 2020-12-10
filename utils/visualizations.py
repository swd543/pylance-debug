# %%
from typing import List
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.pyplot import close
from torchvision import transforms
import numpy as np
from IPython.display import HTML, display_html
import SimpleITK as sitk

def display_animation(img: np.array, save_dir=None):
    fig, _ = plt.subplots(1,1)
    ims = [[plt.imshow(i, animated=True, cmap='gray')] for i in img]
    anim = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=1000)
    plt.close(fig)
    display_html(HTML(anim.to_jshtml()))
    if save_dir is not None:
        anim.save(save_dir, writer='imagemagick', fps=60) 

def combine_segmentations(segs: List[sitk.Image]):
    x=None
    if len(segs) >=2:
        for i in range(len(segs)-1):
            s2=sitk.Cast(segs[i+1],sitk.sitkLabelUInt8)
            if x==None:
                s1=sitk.Cast(segs[i],sitk.sitkLabelUInt8)
                x=sitk.MergeLabelMap(s1,s2)
            else:
                x=sitk.MergeLabelMap(x,s2)
    else:
        x=sitk.Cast(segs[0], sitk.sitkLabelUInt8)
    return x

def window_image(img: sitk.Image, hu_range=None):
    im = img
    if hu_range == None:
        im = sitk.Cast(sitk.IntensityWindowing(im, outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
    else:
        im = sitk.Cast(sitk.IntensityWindowing(im, windowMinimum=hu_range[0], windowMaximum=hu_range[1], outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
    return im

def animate_image_and_segmentations(img: sitk.Image, segs: List[sitk.Image], hu_range=[-200,150], save_dir=None):
    _segs = combine_segmentations(segs)
    _img = window_image(img, hu_range)
    _out = sitk.LabelMapContourOverlay(_segs, _img, opacity=0.9)
    display_animation(sitk.GetArrayFromImage(_out), save_dir)

# %%
if __name__ == "__main__":
    rand = np.random.rand(100,512,512)
    display_animation(rand)