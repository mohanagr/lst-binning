import numba as nb
import numpy as np
import time


@nb.njit(parallel=True)
def set_nans(arr, flags):
    nrows = arr.shape[0]
    nchans = arr.shape[1]
    for i in nb.prange(nrows):
        for j in range(nchans):
            if flags[i, j]:
                arr[i, j] = np.nan


@nb.njit()
def myreduce(arr):
    # newarr=np.sort(arr)
    return np.mean(arr)


@nb.njit(parallel=True)
def myfunc(x, lstbins, bins, result):
    nchans = x.shape[1]
    nbins = len(bins)
    for b in nb.prange(nbins):
        rownums = np.where(lstbins == bins[b])[0]
        temp = np.empty((nchans, len(rownums)), dtype="float64")
        for i, rownum in enumerate(rownums):
            temp[:, i] = x[rownum, :]  # * flags[rownum,:]
            # print(f"rownum {rownum}, temp is", temp)
        for c in range(nchans):
            result[b, c] = np.nanmean(temp[c, :])


@nb.njit()
def myfunc2(x, lstbins, bins, maxnum, result):
    nchans = x.shape[1]
    nrows = x.shape[0]
    nbins = len(bins)
    binmap = np.empty((len(bins), maxnum), dtype="int64")
    bincounts = np.zeros(len(bins), dtype="int64")
    for b in range(nrows):
        lstbin = lstbins[b]
        binmap[lstbin, bincounts[lstbin]] = b
        bincounts[lstbin] += 1
    # print(bincounts)
    # print(binmap)
    for b in nb.prange(nbins):
        # rownums=np.where(lstbins==bins[b])[0]
        lstbin = b
        bincount = bincounts[lstbin]
        # print("looking at lstbin", lstbin, "that has bincount", bincount)
        temp = np.empty((nchans, bincount), dtype="float64")
        for i in range(bincount):
            # print("lstbin", lstbin, "location", binmap[lstbin,i])
            temp[:, i] = x[binmap[lstbin, i], :]  # * flags[rownum,:]
            for c in range(nchans):
                result[b, c] = np.nanmedian(temp[c, :])


@nb.njit(parallel=True)
def myfunc3(x, lstbins, result):
    nchans = x.shape[1]
    nrows = x.shape[0]
    nbins = len(bins)
    idx = np.argsort(lstbins)
    lstbins_sorted = np.sort(lstbins)
    binchanges = np.where(np.diff(lstbins_sorted) != 0)[0]
    nchanges = len(binchanges)
    for b in nb.prange(nchanges):
        # rownums=np.where(lstbins==bins[b])[0]
        jj = binchanges[b]
        lstbin = lstbins_sorted[jj]
        if b == 0:
            bincount = jj + 1
        elif b == nchanges - 1:
            bincount = nrows - jj - 1
        else:
            bincount = binchanges[b + 1] - jj
        temp = np.empty((nchans, bincount), dtype="float64")
        for i in range(bincount):
            for c in range(nchans):
                result[b, c] = np.nanmean(temp[c, :])


if __name__ == "__main__":
    np.set_printoptions(threshold=100000)
    nbins = 5
    nchans = 5
    maxnum = 300
    low = 200
    np.random.seed(42)
    bins = np.arange(nbins, dtype="int64")  # could be anything
    bindist = np.random.randint(
        low, high=maxnum, size=nbins
    )  # how many of each bin type
    # print("bins are", bins, "bindist",bindist)
    x = np.empty((np.sum(bindist), nchans), dtype="float64")
    lstbins = np.empty((np.sum(bindist)), dtype="int64")
    nrows = 0
    for b in bins:
        x[nrows : nrows + bindist[b]] = (
            np.ones((bindist[b], nchans), dtype="float64") * b
        )
        lstbins[nrows : nrows + bindist[b]] = b
        nrows += bindist[b]
    # print("final shape",x.shape)
    # print("lstbins",lstbins)
    # arridx=np.arange(x.shape[0])
    p = np.random.permutation(len(lstbins))
    xreal = x[p, :].copy()
    lstbinsreal = lstbins[p].copy()
    # print("lstbinsreal",lstbinsreal)
    # print("xreal", xreal)
    result = np.empty((nbins, nchans), dtype="float64")
    expected = np.arange(nbins, dtype="float64").reshape(nbins, 1) @ (
        np.ones(nchans).reshape(1, nchans)
    )
    myfunc(xreal, lstbinsreal, bins, result)
    assert np.allclose(result, expected)
    # print("result", result)
    niter = 10
    for i in range(niter):
        t1 = time.time()
        myfunc(xreal, lstbinsreal, bins, result)
        t2 = time.time()
        # print("xreal\n",xreal,"lstbins real",lstbinsreal)
        print("taking", t2 - t1)
