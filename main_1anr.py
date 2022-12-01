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

        fname = "/data/users/c3471tl/rss_regressor/testdata_" + basename

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

        fname = "/data/users/c3471tl/rss_regressor/traindata_" + basename
        dataset = joblib.load(fname)
        print("loaded", fname, "!")

        anum = 7

        traindata = dataset[anum]
        data_idx = (anum * 3) + 1

        xdata = traindata[:, :data_idx]
        ydata = traindata[:, -2:]
        xscaler = MinMaxScaler(feature_range=(-1, 1))
        xscaled = xscaler.fit_transform(xdata)

        # separate data into train and test group
        testratio = 0.3
        valratio = 0.3
        xtmp, xtest, ytmp, ytest = train_test_split(xscaled, ydata, test_size=testratio, shuffle=True, random_state=7)
        xtrain, xval, ytrain, yval = train_test_split(xtmp, ytmp, test_size=valratio, shuffle=False)

        # define baseline mode
        # nodelist = [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800]
        trainrmselist = []
        testrmselist = []

        # for node in nodelist:
        model = Sequential()
        model.add(Dense(512, input_dim=xtrain.shape[1], activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(ytrain.shape[1], activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
        history = model.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=10000, verbose=0, callbacks=[es])
        train_mse = model.evaluate(xtrain, ytrain, verbose=0)
        train_rmse = np.sqrt(train_mse * 2)
        test_mse = model.evaluate(xtest, ytest, verbose=0)
        test_rmse = np.sqrt(test_mse * 2)
        print("train rmse = {:.4f}, test rmse = {:.4f}".format(train_rmse, test_rmse))
        # trainrmselist.append(train_rmse)
        # testrmselist.append(test_rmse)

        print(model.summary())
        model.save("/data/users/c3471tl/rss_regressor/train_model_anum" + str(anum) + ".h5")
        joblib.dump(xscaler, "/data/users/c3471tl/rss_regressor/train_scaler_anum" + str(anum) + ".save")
        print("saved!")

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['train', 'validate'])
        plt.xlabel('epoch')
        plt.ylabel('mse')
        plt.show()

        # plt.plot(epochlist, trainrmselist, 'r-x', label="train")
        # plt.plot(epochlist, testrmselist, 'k-x', label="test")
        # plt.xlabel("Epochs")
        # plt.ylabel("Root mean squared error")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # # save model and scaler
        # model.save("model_r1.0_a5_n900e400b16.h5")
        # scale_fname = "xscaler_r1.0_a5_n900e400b16.save"
        # joblib.dump(xscaler, scale_fname)
        # print("saved model!")

        fname = "/data/users/c3471tl/rss_regressor/testdata_" + basename
        dataset = joblib.load(fname)

        # model trained with noise parameters
        # nodelist = [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800]
        # rangestd = 10 ** (1 / 10)
        # anglestd = 5
        testdata = dataset[anum]

        xdata = testdata[:, :data_idx]
        ydata = testdata[:, -2:]

        # for node in nodelist:
        model = load_model("/data/users/c3471tl/rss_regressor/train_model_anum" + str(anum) + ".h5")
        xscaler = joblib.load("/data/users/c3471tl/rss_regressor/train_scaler_anum" + str(anum) + ".save")
        xscaled = xscaler.transform(xdata)

        apostestx = xscaled[:, :anum]
        apostesty = xscaled[:, anum:(2 * anum)]
        tpostestx = ydata[:, 0]
        tpostesty = ydata[:, 1]

        # deep learning
        mse_dl = model.evaluate(xscaled, ydata, verbose=0)
        rmse_dl = np.sqrt(mse_dl * 2)
        print("rmse dl = {:.4f}".format(rmse_dl))

        # CRLB
        crlb = util.getracrlb(testdata, anum, rangestd, anglestd, gamma=ple)
        print("rmse crlb = {:.4f}".format(crlb))

        # nr-ls
        rmse_nr_ls = estimator.evaluate("nr-ls", testdata, anum)
        print("rmse nr-ls = {:.4f}".format(rmse_nr_ls))

        # nr-ss
        rmse_nr_ss = estimator.evaluate("nr-ss", testdata, anum)
        print("rmse nr-ss = {:.4f}".format(rmse_nr_ss))

        # nrna-ls
        rmse_nrna_ls = estimator.evaluate("nrna-ls", testdata, anum)
        print("rmse nrna-ls = {:.4f}".format(rmse_nrna_ls))

        # nrna-wls
        rmse_nrna_wls = estimator.evaluate("nrna-wls", testdata, anum)
        print("rmse nrna-wls = {:.4f}".format(rmse_nrna_wls))

        # nr1a-ls
        rmse_nr1a_ls = estimator.evaluate("nr1a-ls", testdata, anum)
        print("rmse nr1a-ls = {:.4f}".format(rmse_nr1a_ls))

        # nr1a-ss
        rmse_nr1a_ss = estimator.evaluate("nr1a-ss", testdata, anum)
        print("rmse nr1a-ss = {:.4f}".format(rmse_nr1a_ss))

        # nr1a-minls
        rmse_nr1a_minls = estimator.evaluate("nr1a-minls", testdata, anum)
        print("rmse nr1a-minls = {:.4f}".format(rmse_nr1a_minls))
