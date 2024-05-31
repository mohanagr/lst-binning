import numba as nb
import numpy as np


@nb.njit(parallel=True)
def set_nans(arr, flags):
    nrows = arr.shape[0]
    nchans = arr.shape[1]
    for i in nb.prange(nrows):
        for j in nb.prange(nchans):
            if flags[i, j]:
                arr[i, j] = np.nan


@nb.njit(parallel=True)
def myreduce(arr):
    # newarr=np.sort(arr)
    return np.mean(arr)


# @nb.njit(parallel=True)
# def myfunc(x, lstbins, bins, result):
#    nchans=x.shape[1]
#    nbins = len(bins)
#    for b in nb.prange(nbins):
#        rownums=np.where(lstbins==bins[b])[0]
#        temp=np.empty((nchans,len(rownums)),dtype="float64")
#        for i, rownum in enumerate(rownums):
#            temp[:, i]=x[rownum,:] #* flags[rownum,:]
#            for c in range(nchans):
#                result[b,c]=np.nanmean(temp[c,:])


@nb.njit(parallel=True)
def myfunc(x, lstbins, bins, result):
    """
    Generate an LST binned array from an aggregate of direct data.
    Datatype of the input/output arrays is assumed to be float64.

    Parameters
    ----------
    x : np.ndarray
    INPUT 2D array of raw data (nrows x nchans)
    nchans is the number of channels in the raw data. Typically 2048.
    lstbins : np.ndarray
    1D vector of LST bins corresponding to each row of x
    bins : np.ndarray
    Unique LST bins that the data consists of.
    E.g. np.arange(0,24) for 1-hourly bins.
    result : np.ndarray
    OUTPUT 2D array of LST-binned data ( len(bins) x nchans )

    Returns
    -------
    None
    """
    nchans = x.shape[1]
    nbins = len(bins)
    for b in nb.prange(nbins):
        rownums = np.where(lstbins == bins[b])[0]
        temp = np.empty((nchans, len(rownums)), dtype="float64")
        for i, rownum in enumerate(rownums):
            temp[:, i] = x[rownum, :]  # * flags[rownum,:]
            # print(f"rownum {rownum}, temp is", temp)
        for c in range(nchans):
            result[b, c] = np.nanmedian(temp[c, :])


if __name__ == "__main__":
    myfunc(1, 2, 3, 4)
