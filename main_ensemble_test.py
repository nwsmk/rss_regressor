from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm
import joblib
import util
import estimator


if __name__ == '__main__':

    gendataTrain = False
    gendataTest = False
    trainmodel = True
    anum_list = [3, 4, 5, 6, 7]
    # -----------------------------------------------------------
    # room dimension (meters)
    rx = 50
    ry = 50
    # noise parameters
    ple = 4
    rangestd = 1.0
    anglestd = 5

    basename = "r1.0_a5.save"

    if gendataTrain:

        fname = "/data/users/c3471tl/rss_regressor/traindata_" + basename

        # target
        # -----------------------------------------------------------
        # number of target position patterns per each anchor setting
        tposnum = 1000
        # target positions
        tposx = np.random.uniform(0, rx, (tposnum, 1))
        tposy = np.random.uniform(0, ry, (tposnum, 1))

        # anchor
        # -----------------------------------------------------------
        # number of anchor position patterns
        aposnum = 1000
        # number of maximum anchors in the system
        anummax = 7
        # anchor positions
        aposmaxx = np.random.uniform(0, rx, (aposnum, anummax))
        aposmaxy = np.random.uniform(0, ry, (aposnum, anummax))

        # create the full training dataset
        aposxlist = []
        aposylist = []

        # create anchor positions matrix
        for i in range(aposnum):
            aposx = npm.repmat(aposmaxx[i, :], tposnum, 1)
            aposy = npm.repmat(aposmaxy[i, :], tposnum, 1)
            aposxlist.append(aposx)
            aposylist.append(aposy)
        aposmatx = np.concatenate(aposxlist, axis=0)
        aposmaty = np.concatenate(aposylist, axis=0)

        # create target positions matrix
        tposmatx = npm.repmat(tposx, aposnum, 1)
        tposmaty = npm.repmat(tposy, aposnum, 1)

        # create ranges matrix & noisy ranges matrix
        rangesmat = util.getrangesmat(aposmatx, aposmaty, tposmatx, tposmaty)
        noisyrangesmat = util.getnoisyrangesmat(rangesmat, ple, rangestd)

        # create angles matrix & noisy angles matrix
        anglesmat = util.getanglesmat(aposmatx, aposmaty, tposmatx, tposmaty)
        noisyanglesmat = util.getnoisyanglesmat(anglesmat, anglestd)

        # separate the training dataset into the cases of 3, 4, 5, 6 and 7 anchors
        dataset = {}
        anumlist = [3, 4, 5, 6, 7]
        for anum in anumlist:
            ndata = np.hstack((aposmatx[:, :anum], aposmaty[:, :anum], noisyrangesmat[:, :anum],
                               noisyanglesmat[:, :anum], tposmatx, tposmaty))
            tdata = np.hstack((aposmatx[:, :anum], aposmaty[:, :anum], rangesmat[:, :anum],
                               anglesmat[:, :anum], tposmatx, tposmaty))
            dataset[anum] = np.vstack((ndata, tdata))

        joblib.dump(dataset, fname)
        print("saved train data!")

    elif gendataTest:

        fname = "testdata_" + basename

        # target
        # -----------------------------------------------------------
        # number of target position patterns per each anchor setting
        tposnum = 100
        # target positions
        tposx = np.random.uniform(0, rx, (tposnum, 1))
        tposy = np.random.uniform(0, ry, (tposnum, 1))

        # anchor
        # -----------------------------------------------------------
        # number of anchor position patterns
        aposnum = 100
        # number of maximum anchors in the system
        anummax = 7
        # anchor positions
        aposmaxx = np.random.uniform(0, rx, (aposnum, anummax))
        aposmaxy = np.random.uniform(0, ry, (aposnum, anummax))

        # create the full training dataset
        aposxlist = []
        aposylist = []

        # create anchor positions matrix
        for i in range(aposnum):
            aposx = npm.repmat(aposmaxx[i, :], tposnum, 1)
            aposy = npm.repmat(aposmaxy[i, :], tposnum, 1)
            aposxlist.append(aposx)
            aposylist.append(aposy)
        aposmatx = np.concatenate(aposxlist, axis=0)
        aposmaty = np.concatenate(aposylist, axis=0)

        # create target positions matrix
        tposmatx = npm.repmat(tposx, aposnum, 1)
        tposmaty = npm.repmat(tposy, aposnum, 1)

        # create ranges matrix & noisy ranges matrix
        rangesmat = util.getrangesmat(aposmatx, aposmaty, tposmatx, tposmaty)
        noisyrangesmat = util.getnoisyrangesmat(rangesmat, ple, rangestd)

        # create angles matrix & noisy angles matrix
        anglesmat = util.getanglesmat(aposmatx, aposmaty, tposmatx, tposmaty)
        noisyanglesmat = util.getnoisyanglesmat(anglesmat, anglestd)

        # separate the training dataset into the cases of 3, 4, 5, 6 and 7 anchors
        dataset = {}
        anumlist = [3, 4, 5, 6, 7]
        for anum in anumlist:
            ndata = np.hstack((aposmatx[:, :anum], aposmaty[:, :anum], noisyrangesmat[:, :anum],
                               noisyanglesmat[:, :anum], tposmatx, tposmaty))

            dataset[anum] = ndata

        joblib.dump(dataset, fname)
        print("saved test data!")

    ##################################################################################################
    elif trainmodel:

        anum = 7

        print("ANUM = {} -------------------------------------------------- ".format(anum))
        for count in range(1):

            testfname = "testdata_" + basename
            testdataset = joblib.load(testfname)

            testdata = testdataset[anum]

            xtestdata = testdata[:, :-2]
            ytestdata = testdata[:, -2:]

            # for node in nodelist:
            rmse = 0.0
            num_model = 10
            y_pred = np.zeros(ytestdata.shape)
            for m in range(num_model):
                model = load_model("train_model_anum" + str(anum) + "_c" + str(m) + ".h5")
                xscaler = joblib.load("train_scaler_anum" + str(anum) + "_c" + str(m) + ".save")
                xscaled = xscaler.transform(xtestdata)
                y_pred = y_pred + model.predict(xscaled)

            y_comb = (1 / num_model) * y_pred

            np_est_pos = y_comb
            np_tru_pos = ytestdata
            pos_err = np.sqrt(np.sum(np.square(np_est_pos - np_tru_pos), axis=1))
            pos_err_rmse = np.sqrt(np.mean(np.sum(np.square(np_est_pos - np_tru_pos), axis=1)))
            err50pc = np.percentile(pos_err, 50, interpolation='linear')
            err80pc = np.percentile(pos_err, 80, interpolation='linear')
            print("dl rmse = {:.2f}, median = {:.2f}, 80th% = {:.2f}".format(pos_err_rmse, err50pc, err80pc))

            # # CRLB
            # crlb = util.getracrlb(testdata, anum, rangestd, anglestd, gamma=ple)
            # print("rmse crlb = {:.4f}".format(crlb))
            #
            # # nr-ls
            # rmse_nr_ls = estimator.evaluate("nr-ls", testdata, anum)
            # print("rmse nr-ls = {:.4f}".format(rmse_nr_ls))
            #
            # # nr-ss
            # rmse_nr_ss = estimator.evaluate("nr-ss", testdata, anum)
            # print("rmse nr-ss = {:.4f}".format(rmse_nr_ss))
            #
            # nrna-ls
            rmse_nrna_ls = estimator.evaluate("nrna-ls", testdata, anum)
            print("rmse nrna-ls = {:.2f}".format(rmse_nrna_ls))
            #
            # nrna-wls
            rmse_nrna_wls = estimator.evaluate("nrna-wls", testdata, anum)
            print("rmse nrna-wls = {:.2f}".format(rmse_nrna_wls))
            #
            # # nr1a-ls
            # rmse_nr1a_ls = estimator.evaluate("nr1a-ls", testdata, anum)
            # print("rmse nr1a-ls = {:.4f}".format(rmse_nr1a_ls))
            #
            # # nr1a-ss
            # rmse_nr1a_ss = estimator.evaluate("nr1a-ss", testdata, anum)
            # print("rmse nr1a-ss = {:.4f}".format(rmse_nr1a_ss))
            #
            # # nr1a-minls
            # rmse_nr1a_minls = estimator.evaluate("nr1a-minls", testdata, anum)
            # print("rmse nr1a-minls = {:.4f}".format(rmse_nr1a_minls))
