from unet.unet_model import UNet
from skimage import metrics
import matplotlib.pyplot as plt
from utils.dataLoader import dataLoader
import utils.supportingFunctions as sf
import numpy as np
from torch.nn.functional import normalize
import torch


def recoverTest(imgNum):

    trnOrg, trnAtb, _, mask = dataLoader()
    # trnOrg, trnAtb, _, mask = sf.getData('training')
    trnOrg, trnAtb = sf.c2r(trnOrg), sf.c2r(trnAtb)

    model = UNet(2, 2)
    model.load_state_dict(torch.load('/Users/henry/PycharmProjects/MRI/checkpoints/epoch_20.pkl'))
    model.train(False)

    refImg = torch.tensor(trnOrg[imgNum]).squeeze().permute(2, 0, 1)[None, :]
    img = torch.tensor(trnAtb[imgNum]).squeeze().permute(2, 0, 1)[None, :]

    predImg = model(img)

    normOrg = sf.normalize01(np.abs(sf.r2c(refImg.detach().squeeze().permute(1, 2, 0).numpy())))
    normAtb = sf.normalize01(np.abs(sf.r2c(img.detach().squeeze().permute(1, 2, 0).numpy())))
    normRec = sf.normalize01(np.abs(sf.r2c(predImg.detach().squeeze().permute(1, 2, 0).numpy())))

    PSNR = sf.myPSNR(normOrg, normRec)
    SSIM = metrics.structural_similarity(normOrg, normRec)
    print(PSNR, SSIM)

    PSNR = sf.myPSNR(normOrg, normAtb)
    SSIM = metrics.structural_similarity(normOrg, normAtb)
    print(PSNR, SSIM)

    # plot = lambda x: plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, .8))
    # plt.axis('off')
    # plot(normOrg)
    # plt.savefig('/Users/henry/Downloads/'+str(imgNum)+'_ORG.png', bbox_inches='tight', pad_inches=-0.1)
    # plot(normAtb)
    # plt.savefig('/Users/henry/Downloads/' + str(imgNum) + '_DIST.png', bbox_inches='tight', pad_inches=-0.1)
    # plot(normRec)
    # plt.savefig('/Users/henry/Downloads/' + str(imgNum) + '_OUT.png', bbox_inches='tight', pad_inches=-0.1)
    # plot(mask[imgNum])
    # plt.savefig('/Users/henry/Downloads/' + str(imgNum) + '_MASK.png', bbox_inches='tight', pad_inches=-0.1)

if __name__ == '__main__':
    recoverTest(203)