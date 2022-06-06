import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import cv2

from skimage.segmentation import expand_labels

maxIter = 1000
nChannels = 100
nConv = 2
lr = 0.1
stepSizeSim, stepSizeCon, stepSizeScr = 1, 1, 0.5


class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv - 1):
            self.conv2.append(nn.Conv2d(nChannels, nChannels, kernel_size=(3, 3), stride=(1, 1), padding=1))
            self.bn2.append(nn.BatchNorm2d(nChannels))
        self.conv3 = nn.Conv2d(nChannels, nChannels, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(nChannels)

    def forward(self, x):
        x = self.conv1(x)
        x = func.relu(x)
        x = self.bn1(x)
        for i in range(nConv - 1):
            x = self.conv2[i](x)
            x = func.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


def segment(img, max_labels, max_segment_id):

    orig_dims = (img.shape[1], img.shape[0])

    dim = (int(orig_dims[0] / 5), int(orig_dims[1] / 5))

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_NEAREST)

    use_cuda = torch.cuda.is_available()

    data = torch.from_numpy(np.array([img.transpose((2, 0, 1)).astype('float32') / 255.]))
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    model = MyNet(data.size(1))
    if use_cuda:
        model.cuda()
    model.train()

    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(size_average=True)
    loss_hpz = torch.nn.L1Loss(size_average=True)

    h_py_target = torch.zeros(img.shape[0] - 1, img.shape[1], nChannels)
    h_pz_target = torch.zeros(img.shape[0], img.shape[1] - 1, nChannels)
    if use_cuda:
        h_py_target = h_py_target.cuda()
        h_pz_target = h_pz_target.cuda()

    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    n_labels = -1

    for batch_idx in range(maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, nChannels)

        output_hp = output.reshape((img.shape[0], img.shape[1], nChannels))
        h_py = output_hp[1:, :, :] - output_hp[0:-1, :, :]
        h_pz = output_hp[:, 1:, :] - output_hp[:, 0:-1, :]
        lhpy = loss_hpy(h_py, h_py_target)
        lhpz = loss_hpz(h_pz, h_pz_target)

        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        n_labels = len(np.unique(im_target))

        # loss
        loss = stepSizeSim * loss_fn(output, target) + stepSizeCon * (lhpy + lhpz)

        print(batch_idx, '/', maxIter, '|', ' label num :', n_labels, ' | loss :', loss.item())

        loss.backward()
        optimizer.step()

        if n_labels <= max_labels:
            break

    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, nChannels)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target = im_target.reshape((img.shape[0], img.shape[1])).astype(np.uint8)

    max_segment_id += 1

    # reduce to maxLabels if there are too many
    if n_labels > max_labels:
        unique_vals, counts = np.unique(im_target, return_counts=True)
        clusters = np.column_stack([unique_vals, counts])
        clusters_sorted = clusters[np.argsort(clusters[:, 1])]

        i = -1
        for elem in reversed(clusters_sorted):

            i = i + 1

            if i < max_labels:
                continue

            im_target[im_target == elem[0]] = 0

        # fill the background with the surrounding pixels
        im_target = expand_labels(im_target, distance=1000)

    unique_vals = np.unique(im_target)

    for elem in unique_vals:
        im_target[im_target == elem] = max_segment_id
        max_segment_id += 1

    im_target = cv2.resize(im_target, dsize=orig_dims, interpolation=cv2.INTER_NEAREST)

    #im_target = np.transpose(im_target)
    #im_target = np.flip(im_target, 1)
    #im_target = np.flip(im_target, 0)
    return im_target
