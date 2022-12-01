import numpy as np
import numpy.linalg as npl


# n-RSS least square
def nr_ls(aposx, aposy, ranges):
    # H matrix
    hx = aposx[1:] - aposx[0]
    hy = aposy[1:] - aposy[0]
    h = np.vstack((hx, hy)).T
    # b matrix
    ksq = np.square(aposx) + np.square(aposy)
    rsq = np.square(ranges)
    b = ((1 / 2) * (ksq[1:] - ksq[0] - rsq[1:] + rsq[0])).T
    estpos = npl.inv(h.T.dot(h)).dot(h.T).dot(b)
    return estpos[0], estpos[1]


# n-RSS subspace
def nr_ss(aposx, aposy, ranges):
    # A matrix
    a = np.vstack((aposx, aposy)).T
    # D matrix
    alen = len(aposx)
    d = np.zeros((alen, alen))
    for m in range(alen):
        for n in range(alen):
            d[m, n] = (1 / 2) * ((ranges[m] ** 2) + (ranges[n] ** 2) -
                                 (((aposx[m] - aposx[n]) ** 2) + ((aposy[m] - aposy[n]) ** 2)))

    # eigenvalue decomposition
    evals, evecs = npl.eig(d)
    evalssort = np.argsort(evals)
    evalsid = evalssort[-2:]
    us = np.real(evecs[:, evalsid])
    un = np.identity(alen) - (us.dot(us.T))
    ones = np.ones((alen, 1))
    estpos = (ones.T.dot(un).dot(a)) / (ones.T.dot(un).dot(ones))
    return estpos[0, 0], estpos[0, 1]


# n-RSS & n-AOA least square
def nrna_ls(aposx, aposy, ranges, angles):
    # S matrix
    alen = len(aposx)
    ones = np.ones((alen, 1))
    zeros = np.zeros((alen, 1))
    s1 = np.hstack((ones, zeros))
    s2 = np.hstack((zeros, ones))
    s = np.vstack((s1, s2))
    ux = aposx + np.multiply(ranges, np.cos(angles))
    uy = aposy + np.multiply(ranges, np.sin(angles))
    u = np.hstack((ux, uy)).T
    estpos = npl.inv(s.T.dot(s)).dot(s.T).dot(u)
    return estpos[0], estpos[1]


# n-RSS & n-AOA weighted least square
def nrna_wls(aposx, aposy, ranges, angles):
    # S matrix
    alen = len(aposx)
    ones = np.ones((alen, 1))
    zeros = np.zeros((alen, 1))
    s1 = np.hstack((ones, zeros))
    s2 = np.hstack((zeros, ones))
    s = np.vstack((s1, s2))
    ux = aposx + np.multiply(ranges, np.cos(angles))
    uy = aposy + np.multiply(ranges, np.sin(angles))
    u = np.hstack((ux, uy)).T
    sumr = np.sum(ranges)
    w1 = np.sqrt(1 - ((1 / sumr) * ranges))
    w = np.kron(np.identity(2), np.diag(w1))
    estpos = npl.inv(s.T.dot(w.T).dot(s)).dot(s.T).dot(w.T).dot(u)
    return estpos[0], estpos[1]


# n-RSS & 1-AOA least square
def nr1a_ls(aposx, aposy, ranges, angles):
    aposrefx = aposx[0]
    aposrefy = aposy[0]
    rangesref = ranges[0]
    anglesref = angles[0]
    aposv1x = aposrefx + (rangesref * np.cos(anglesref))
    aposv1y = 0
    rangesv1 = rangesref * np.sin(anglesref)
    aposv2x = 0
    aposv2y = aposrefy + (rangesref * np.sin(anglesref))
    rangesv2 = rangesref * np.cos(anglesref)
    newaposx = np.hstack((aposx, aposv1x, aposv2x))
    newaposy = np.hstack((aposy, aposv1y, aposv2y))
    newranges = np.hstack((ranges, rangesv1, rangesv2))
    return nr_ls(newaposx, newaposy, newranges)


# n-RSS & 1-AOA subspace
def nr1a_ss(aposx, aposy, ranges, angles):
    aposrefx = aposx[0]
    aposrefy = aposy[0]
    rangesref = ranges[0]
    anglesref = angles[0]

    aposv1x = aposrefx + (rangesref * np.cos(anglesref))
    aposv1y = aposrefy + 0
    rangesv1 = rangesref * np.sin(anglesref)
    aposv2x = aposrefx + 0
    aposv2y = aposrefy + (rangesref * np.sin(anglesref))

    rangesv2 = rangesref * np.cos(anglesref)
    newaposx = np.hstack((aposx, aposv1x, aposv2x))
    newaposy = np.hstack((aposy, aposv1y, aposv2y))
    newranges = np.hstack((ranges, rangesv1, rangesv2))
    return nr_ss(newaposx, newaposy, newranges)


# n-RSS & 1-AOA least square minref
def nr1a_minls(aposx, aposy, ranges, angles):
    sortid = np.argsort(ranges)
    sortaposx = aposx[sortid]
    sortaposy = aposy[sortid]
    sortranges = ranges[sortid]
    sortangles = angles[sortid]
    aposrefx = sortaposx[0]
    aposrefy = sortaposy[0]
    rangesref = sortranges[0]
    anglesref = sortangles[0]

    aposv1x = aposrefx + (rangesref * np.cos(anglesref))
    aposv1y = aposrefy + 0
    rangesv1 = rangesref * np.sin(anglesref)

    aposv2x = aposrefx + 0
    aposv2y = aposrefy + (rangesref * np.sin(anglesref))
    rangesv2 = rangesref * np.cos(anglesref)

    newaposx = np.hstack((sortaposx, aposv1x, aposv2x))
    newaposy = np.hstack((sortaposy, aposv1y, aposv2y))
    newranges = np.hstack((sortranges, rangesv1, rangesv2))
    return nr_ls(newaposx, newaposy, newranges)


def evaluate(algo, testdata, anum):
    nrow, ncol = testdata.shape
    sqerr = 0

    if algo == "nr-ls":
        for data in testdata:
            # extract data for traditional method
            aposx = data[:anum]
            aposy = data[anum:(2 * anum)]
            ranges = data[(2 * anum):(3 * anum)]
            angles = data[(3 * anum):(4 * anum)]
            tposx = data[(4 * anum)]
            tposy = data[(4 * anum) + 1]

            estposx, estposy = nr_ls(aposx, aposy, ranges)
            sqerr = sqerr + ((tposx - estposx) ** 2) + ((tposy - estposy) ** 2)

    elif algo == "nr-ss":
        for data in testdata:
            # extract data for traditional method
            aposx = data[:anum]
            aposy = data[anum:(2 * anum)]
            ranges = data[(2 * anum):(3 * anum)]
            angles = data[(3 * anum):(4 * anum)]
            tposx = data[(4 * anum)]
            tposy = data[(4 * anum) + 1]

            estposx, estposy = nr_ss(aposx, aposy, ranges)
            sqerr = sqerr + ((tposx - estposx) ** 2) + ((tposy - estposy) ** 2)

    elif algo == "nrna-ls":
        for data in testdata:
            # extract data for traditional method
            aposx = data[:anum]
            aposy = data[anum:(2 * anum)]
            ranges = data[(2 * anum):(3 * anum)]
            angles = data[(3 * anum):(4 * anum)]
            tposx = data[(4 * anum)]
            tposy = data[(4 * anum) + 1]

            estposx, estposy = nrna_ls(aposx, aposy, ranges, angles)
            sqerr = sqerr + ((tposx - estposx) ** 2) + ((tposy - estposy) ** 2)

    elif algo == "nrna-wls":
        for data in testdata:
            # extract data for traditional method
            aposx = data[:anum]
            aposy = data[anum:(2 * anum)]
            ranges = data[(2 * anum):(3 * anum)]
            angles = data[(3 * anum):(4 * anum)]
            tposx = data[(4 * anum)]
            tposy = data[(4 * anum) + 1]

            estposx, estposy = nrna_wls(aposx, aposy, ranges, angles)
            sqerr = sqerr + ((tposx - estposx) ** 2) + ((tposy - estposy) ** 2)

    elif algo == "nr1a-ls":
        for data in testdata:
            # extract data for traditional method
            aposx = data[:anum]
            aposy = data[anum:(2 * anum)]
            ranges = data[(2 * anum):(3 * anum)]
            angles = data[(3 * anum):(4 * anum)]
            tposx = data[(4 * anum)]
            tposy = data[(4 * anum) + 1]

            estposx, estposy = nr1a_ls(aposx, aposy, ranges, angles)
            sqerr = sqerr + ((tposx - estposx) ** 2) + ((tposy - estposy) ** 2)

    elif algo == "nr1a-ss":
        for data in testdata:
            # extract data for traditional method
            aposx = data[:anum]
            aposy = data[anum:(2 * anum)]
            ranges = data[(2 * anum):(3 * anum)]
            angles = data[(3 * anum):(4 * anum)]
            tposx = data[(4 * anum)]
            tposy = data[(4 * anum) + 1]

            estposx, estposy = nr1a_ss(aposx, aposy, ranges, angles)
            sqerr = sqerr + ((tposx - estposx) ** 2) + ((tposy - estposy) ** 2)

    elif algo == "nr1a-minls":
        for data in testdata:
            # extract data for traditional method
            aposx = data[:anum]
            aposy = data[anum:(2 * anum)]
            ranges = data[(2 * anum):(3 * anum)]
            angles = data[(3 * anum):(4 * anum)]
            tposx = data[(4 * anum)]
            tposy = data[(4 * anum) + 1]

            estposx, estposy = nr1a_minls(aposx, aposy, ranges, angles)
            sqerr = sqerr + ((tposx - estposx) ** 2) + ((tposy - estposy) ** 2)

    rmse = np.sqrt(sqerr / nrow)
    return rmse
