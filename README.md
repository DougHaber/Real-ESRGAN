# Real-ESRGAN

## WARNING: This is an unmaintained fork

The repository [DougHaber/Real-ESRGAN](https://github.com/DougHaber/Real-ESRGAN) is a fork of [ai-forever/Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN) which is a fork of [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

The ai-forever / sberbank-ai version added in an improved Python interface that I wanted to use.  It hadn't been updated in a couple of years, and all usage was triggering a couple Torch warnings. The changes in this repository fix those warnings.

These changes have been made available as a [pull request](https://github.com/ai-forever/Real-ESRGAN/pull/34) to the upstream as well.

As this is unmaintained, usage is not recommended, but if you feel the need, the correct URL is:

```bash
pip install git+https://github.com/DougHaber/Real-ESRGAN.git
```

In the event that the PR is accepted, this fork will likely be deleted.

All content below this section is unmodified from the upstream repository.

## Overview


PyTorch implementation of a Real-ESRGAN model trained on custom dataset. This model shows better results on faces compared to the original version. It is also easier to integrate this model into your projects.

> This is not an official implementation. We partially use code from the [original repository](https://github.com/xinntao/Real-ESRGAN)

Real-ESRGAN is an upgraded [ESRGAN](https://arxiv.org/abs/1809.00219) trained with pure synthetic data is capable of enhancing details while removing annoying artifacts for common real-world images. 

You can try it in [google colab](https://colab.research.google.com/drive/1YlWt--P9w25JUs8bHBOuf8GcMkx-hocP?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YlWt--P9w25JUs8bHBOuf8GcMkx-hocP?usp=sharing)

- [Paper (Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data)](https://arxiv.org/abs/2107.10833)
- [Original implementation](https://github.com/xinntao/Real-ESRGAN)
- [Huggingface 🤗](https://huggingface.co/sberbank-ai/Real-ESRGAN)

### Installation

```bash
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
```

### Usage

---

Basic usage:

```python
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_to_image = 'inputs/lr_image.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('results/sr_image.png')
```

### Examples

---

Low quality image:

![](inputs/lr_image.png)

Real-ESRGAN result:

![](results/sr_image.png)

---

Low quality image:

![](inputs/lr_face.png)

Real-ESRGAN result:

![](results/sr_face.png)

---

Low quality image:

![](inputs/lr_lion.png)

Real-ESRGAN result:

![](results/sr_lion.png)
