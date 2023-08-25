import utils.supportingFunctions as sf
from skimage import metrics
import numpy as np

def dataLoader():

    refImg, _, _, mask = sf.getData("training")

    mask = np.fft.fftshift(mask)

    f = np.fft.fft2(refImg)  # 快速傅里叶变换算法得到频率分布
    fshift = np.fft.fftshift(f)  # 移频，默认结果中心点位置是在左上角，转移到中间位置
    a_fshift = np.log(np.abs(fshift))  # 取振幅
    ph_fshift = np.angle(fshift)  # 取相位
    # 逆变换--利用欧拉公式 cos(x) + isin(x) ⇔ a + ib
    # 将振幅和相位整合成复数的实部和虚部，即整合成频域数据
    s_real = np.exp(a_fshift) * np.cos(ph_fshift)  # 整合复数的实部
    s_imag = np.exp(a_fshift) * np.sin(ph_fshift)  # 整合复数的虚部
    s = np.zeros([360, 256, 232], dtype=np.complex64)  # 指定数据类型为复数
    s.real = np.array(s_real)
    s.imag = np.array(s_imag)
    ps = mask * s
    fshift = np.fft.ifftshift(ps)  # 对整合好的频域数据进行逆变换
    img = np.fft.ifft2(fshift)
    # 出来的是复数，取绝对值，转化成实数
    # img = np.abs(img)

    return refImg, img, '-', mask

if __name__ == '__main__':
    refImg, img, _, mask = dataLoader()
    print(refImg.shape, img.shape)
    print(refImg.dtype, img.dtype)

    PSNR = []
    SSIM = []
    for i in range(360):
        org = sf.normalize01(np.abs(refImg[i]))
        atb = sf.normalize01(np.abs(img[i]))
        PSNR.append(metrics.peak_signal_noise_ratio(org, atb))
        SSIM.append(metrics.structural_similarity(org, atb))

    print(sum(PSNR)/len(PSNR), sum(SSIM)/len(SSIM))

    # data = []
    # plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, .8))
    #
    # trnOrg, trnAtb, trnCsm, TrnMask = generateData()
    #
    # trnOrg, trnAtb = sf.c2r(trnOrg), sf.c2r(trnAtb)
    #
    # print(trnOrg.shape, trnAtb.shape)

    # normOrg = sf.normalize01(np.abs(trnOrg))
    # normAtb = sf.normalize01(np.abs(sf.r2c(trnAtb)))
    #
    # print(normOrg.shape, normAtb.shape)

    # normOrg = sf.normalize01(np.abs(trnOrg))
    # normAtb = sf.normalize01(np.abs(sf.r2c(trnAtb)))
    #
    # print(normOrg.shape, normAtb.shape)
    #
    # plot(np.abs(normAtb))
    # plt.show()

    # x, x1 = generateData()
    # for i in range(0, 164):
    #     data.append([x[i], x1[i]])
    # for index, [x, x1] in enumerate(data):
    #     plt.axis('off')
    #     plot(x[0])
    #     plt.savefig('/Users/henry/Downloads/REAL/'+str(index)+'_REF.png', bbox_inches='tight', pad_inches=-0.1)
    #     plot(x1[0])
    #     plt.savefig('/Users/henry/Downloads/REAL/'+str(index)+'.png', bbox_inches='tight', pad_inches=-0.1)
    #     plot(x[1])
    #     plt.savefig('/Users/henry/Downloads/IMAG/'+str(index)+'_REF.png', bbox_inches='tight', pad_inches=-0.1)
    #     plot(x1[1])
    #     plt.savefig('/Users/henry/Downloads/IMAG/' + str(index) + '.png', bbox_inches='tight', pad_inches=-0.1)