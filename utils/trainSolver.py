import random
import torch
from unet.unet_model import UNet
from skimage import metrics
from utils.dataLoader import dataLoader
import numpy as np
import utils.supportingFunctions as sf
import matplotlib.pyplot as plt
import os
import sys


class TrainSolver(object):
    """Solver for training and testing"""

    def __init__(self, config):

        self.cnn = UNet(2, 2)
        self.cnn.train(True)
        self.cnn.load_state_dict(torch.load('/Users/henry/PycharmProjects/MRI/checkpoints/epoch_31.pkl'))

        self.loss = torch.nn.MSELoss()
        self.epochs = config.epochs
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        self.test_index = config.test_index

        self.save_dir = config.save_pkl_dir

        paras = [{'params': self.cnn.parameters(), 'lr': self.lr * self.lrratio}]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        trnOrg, trnAtb, _, trnMask = dataLoader()
        trnOrg, trnAtb = sf.c2r(trnOrg), sf.c2r(trnAtb)
        # shape = (360, 256, 232, 2), (360, 256, 232, 2); dtype = float32

        self.trainData = []
        self.testData = []
        for i in self.test_index:
            self.testData.append([trnOrg[i], trnAtb[i], i])
        for i in range(0, 360):
            if i not in self.test_index:
                self.trainData.append([trnOrg[i], trnAtb[i], i])
        random.shuffle(self.trainData)

    def train(self):
        """Training"""
        bestLoss = sys.maxsize

        for t in range(self.epochs):
            print("\nBegin Epoch: {}".format(t + 1))
            epochLoss = []

            for i, [refImg, img, index] in enumerate(self.trainData):
                img = torch.tensor(img)
                img = img.squeeze().permute(2,0,1)
                img = img[None, :]
                refImg = torch.tensor(refImg)
                refImg = refImg.squeeze().permute(2,0,1)
                refImg = refImg[None, :]
                # shape = (1, 2, 256, 232), (1, 2, 256, 232)

                self.solver.zero_grad()
                predImg = self.cnn(img)  # 'paras' contains the network weights conveyed to target network
                loss = self.loss(refImg, predImg)
                epochLoss.append(loss.item())

                loss.backward()
                self.solver.step()

                if i % 10 == 0:
                    print(i, sum(epochLoss) / len(epochLoss))

            print('Complete Epoch: {}'.format(t + 1))
            print('TrainLoss: {}'.format(sum(epochLoss) / len(epochLoss)))
            
            testLoss, testPSNR, testSSIM = self.test(self.testData)
            print('TestLoss: {}, testPSNR: {}, testSSIM: {}'.format(testLoss, testPSNR, testSSIM))

            if testLoss < bestLoss:
                bestLoss = testLoss
            savePath = os.path.join(self.save_dir, 'epoch_{}.pkl'.format(t + 32))
            torch.save(self.cnn.state_dict(), savePath)
            print("model {} saved!".format(savePath))
            
            # Update optimizer
            lr = self.lr / pow(10, (t // 8))
            if t > 2:
                self.lrratio = 5
            elif t > 4:
                self.lrratio = 1
            paras = [{'params': self.cnn.parameters(), 'lr': lr * self.lrratio}]
            self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        print('Best test l1 loss %f' % bestLoss)
        return bestLoss

    def test(self, data):
        """Testing"""
        self.cnn.train(False)
        loss = []
        psnr = []
        ssim = []

        for [refImg, img, index] in data:
            img = torch.tensor(img)
            img = img.squeeze().permute(2, 0, 1)
            img = img[None, :]
            refImg = torch.tensor(refImg)
            refImg = refImg.squeeze().permute(2, 0, 1)
            refImg = refImg[None, :]
            # shape = (1, 2, 256, 232), (1, 2, 256, 232)

            predImg = self.cnn(img)
            loss.append(float(self.loss(predImg, refImg)))

            normOrg = sf.normalize01(np.abs(sf.r2c(refImg.detach().squeeze().permute(1,2,0).numpy())))
            # normAtb = sf.normalize01(np.abs(sf.r2c(img.detach().squeeze().permute(1,2,0).numpy())))
            normRec = sf.normalize01(np.abs(sf.r2c(predImg.detach().squeeze().permute(1,2,0).numpy())))

            psnrRec = sf.myPSNR(normOrg, normRec)
            ssimRec = metrics.structural_similarity(normOrg, normRec)
            psnr.append(psnrRec)
            ssim.append(ssimRec)

            # plot = lambda x: plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, .8))
            # plt.axis("off")
            # plot(normOrg)
            # plt.savefig('/Users/henry/Downloads/MRI_OUT/'+str(index)+'_REF.png', bbox_inches='tight', pad_inches=-0.1)
            # plot(normAtb)
            # plt.savefig('/Users/henry/Downloads/MRI_OUT/'+str(index)+'.png', bbox_inches='tight', pad_inches=-0.1)
            # plot(normRec)
            # plt.savefig('/Users/henry/Downloads/MRI_OUT/'+str(index)+'_PRED.png', bbox_inches='tight', pad_inches=-0.1)

        self.cnn.train(True)
        return sum(loss) / len(loss), sum(psnr) / len(psnr), sum(ssim) / len(ssim)
