import numpy as np
import numpy.linalg as npl


# create ranges matrix
def getrangesmat(aposmatx, aposmaty, tposmatx, tposmaty):
    rangesmat = np.sqrt(np.square(aposmatx - tposmatx) + np.square(aposmaty - tposmaty))
    return rangesmat


# create noisy ranges matrix
def getnoisyrangesmat(rangesmat, ple, noisestd):
    nrow, ncol = rangesmat.shape
    noisesmat = noisestd * np.random.randn(nrow, ncol)
    noisyrangesmat = np.power(10, (10 * ple * np.log10(rangesmat) + noisesmat) / (10 * ple))
    return noisyrangesmat


# create angles matrix
def getanglesmat(aposmatx, aposmaty, tposmatx, tposmaty):
    anglesmat = np.arctan2((tposmaty - aposmaty), (tposmatx - aposmatx)) % (2 * np.pi)
    return anglesmat


# create noisy angles matrix
def getnoisyanglesmat(anglesmat, noisestd):
    nrow, ncol = anglesmat.shape
    noisesmat = noisestd * np.random.randn(nrow, ncol)
    noisyanglesmat = np.radians(np.degrees(anglesmat) + noisesmat)
    return noisyanglesmat


# calculate the CRLB of hybrid RSS/AOA
def getracrlb(testdata, anum, rangestd, anglestd, gamma):

    aposmatx = testdata[:, :anum]
    aposmaty = testdata[:, anum:(2 * anum)]
    tposmatx = testdata[:, -2]
    tposmaty = testdata[:, -1]

    numsample = testdata.shape[0]
    rangevar = rangestd ** 2
    anglevar = ((np.pi / 180) ** 2) * (anglestd ** 2)
    eta = (10 * gamma) / np.log(10)

    crlblist = []
    for i in range(numsample):
        aposx = aposmatx[i, :]
        aposy = aposmaty[i, :]
        tposx = np.reshape(tposmatx[i], (1, 1))
        tposy = np.reshape(tposmaty[i], (1, 1))

        rangesmat = getrangesmat(aposx, aposy, tposx, tposy)

        # calculate FIM
        dfx_r = -1 * eta * (tposx - aposx) / (rangesmat ** 2)
        dfx_a = -1 * (tposy - aposy) / (rangesmat ** 2)
        dfx = np.hstack((dfx_r, dfx_a))
        dfy_r = -1 * eta * (tposy - aposy) / (rangesmat ** 2)
        dfy_a = (tposx - aposx) / (rangesmat ** 2)
        dfy = np.hstack((dfy_r, dfy_a))

        rangevars = rangevar * np.ones((dfx_r.shape[1], 1))
        anglevars = anglevar * np.ones((dfx_a.shape[1], 1))
        varlist = np.vstack((rangevars, anglevars))
        varlist = np.reshape(varlist, (varlist.shape[0], ))
        c = np.diag(varlist)
        dfx = dfx.T
        dfy = dfy.T

        fim = np.zeros((2, 2))
        fim[0, 0] = dfx.T.dot(npl.inv(c)).dot(dfx)
        fim[0, 1] = dfx.T.dot(npl.inv(c)).dot(dfy)
        fim[1, 0] = dfy.T.dot(npl.inv(c)).dot(dfx)
        fim[1, 1] = dfy.T.dot(npl.inv(c)).dot(dfy)
        invfim = npl.inv(fim)

        crlb = np.trace(invfim) / 2
        crlblist.append(crlb)

    avg_crlb = np.sqrt(sum(crlblist) / numsample)

    return avg_crlb