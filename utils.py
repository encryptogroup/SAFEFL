import torch


def MMD(x, y, device):
    """Empirical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.
       Taken and adapted from https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook by
       Onur Tunali

       We use the Gaussian kernel
    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        device: device where computation is performed
    """

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    # RBF or Gaussian kernel
    bandwidth_range = [10, 15, 20, 50]
    for a in bandwidth_range:
        XX += torch.exp(-0.5 * dxx / a)
        YY += torch.exp(-0.5 * dyy / a)
        XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)
