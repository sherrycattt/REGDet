import numpy as np
import torch

data = torch.load('weights/vgg/pyramidbox.pth')
conv1 = np.array(data['vgg.0.weight'].cpu())
inflated_conv1 = np.tile(conv1, (1, 4, 1, 1))
inflated_conv1 = 1 / 4. * inflated_conv1
data['vgg.0.weight'] = torch.from_numpy(inflated_conv1)

torch.save(data, 'weights/vgg/pyramidbox_inflated.pth')
