import numpy as np
from matplotlib import pyplot as plt
import sys

sys.path.insert(0, "/home/s/sievers/mohanagr/")
from albatros_analysis.src.utils import baseband_utils as butils
import numba as nb
from astropy.time import Time
from astropy.coordinates import EarthLocation
from scio import scio
import datetime
import os
import time
import rfi, tools
import multiprocessing as mp


def unix2jd(ctime):
    return ctime / 86400 + 2440587.5


def read_and_filter(fname):
    file = scio.read(fname)
    return {"data": file, "mask": rfi.get_rfi_occupancy(file)}


str2time = lambda tstr: int(datetime.datetime.strptime(tstr, "%Y%m%d").timestamp())

if __name__ == "__main__":
    dtsts = ["20231110", "20230527"]  # start times
    dtens = ["20240320", "20230630"]  # end times
    paths = [
        "/scratch/s/sievers/mohanagr/uapishka_aug2023_mar2024_sd05/data_auto_cross/",
        "/project/s/sievers/albatros/uapishka/202305/data_auto_cross/snap8",
    ]  # corresponding paths for each start-end pair
    # tags=["pol00", "pol01r","pol01i"] #file types to load
    tags = ["pol00", "pol11"]
    nbins = 1440  # number of LST bins
    dH = 24 / nbins
    nchans = 2048
    location = "uapishka"
    fnames = []
    for datebegin, dateend, path in zip(dtsts, dtens, paths):
        print("starting for files between", datebegin, dateend)
        fnames.extend(
            butils.time2fnames(str2time(datebegin), str2time(dateend), path, "d")
        )
    valid_fnames = fnames.copy()
    for fname in fnames:
        hh = datetime.datetime.fromtimestamp(
            butils.get_tstamp_from_filename(fname)
        ).hour
        month = datetime.datetime.fromtimestamp(
            butils.get_tstamp_from_filename(fname)
        ).month
        # Remove daytime data to not have solar RFI
        if month > 3 and month < 10:  # rough summer daytime
            if hh < 22 and hh > 4:
                valid_fnames.remove(fname)
        else:
            if hh < 19 and hh > 5:  # rough winter daytime
                valid_fnames.remove(fname)
    print(f"found {len(valid_fnames)} valid files")
    for tag in tags:
        print(f"reading {tag}")
        new_fnames = [os.path.join(ff, tag + ".scio.bz2") for ff in valid_fnames]
        large_arr = np.zeros((numrows, 2048), dtype="float64")
        # exit(1)
        if tag in ["pol00"]:
            print(f"starting rfi filtering using {tag}...")
            t1 = time.time()
            with mp.Pool(os.cpu_count() // 2) as p:
                filt_data = list(p.map(read_and_filter, new_fnames))
            t2 = time.time()
            print("parallel filter time", t2 - t1)
            numrows = 0
            t1 = time.time()
            for obj in filt_data:
                numrows += obj["data"].shape[0]  # determine total number of rows
            t2 = time.time()
            print("time taken to count rows", t2 - t1)
            print("numrows = ", numrows, "num obj", len(filt_data))
            ctime_arr = np.zeros(numrows, dtype="float64")
            large_arr_mask = np.zeros((numrows, 2048), dtype="bool")
            curidx = 0
            t1 = time.time()
            # go through each loaded file mask and fill the big mask array
            for fnum, obj in enumerate(filt_data):
                ss = obj["data"].shape[0]
                tt = butils.get_tstamp_from_filename(valid_fnames[fnum])
                large_arr_mask[curidx : curidx + ss, :] = obj["mask"]
                large_arr[curidx : curidx + ss, :] = obj["data"]
                ctime_arr[curidx : curidx + ss] = (
                    tt + np.arange(ss) * 6.44
                )  # acclen 393216 = 6.44s. Each row of direct data is 6.44s long.
                curidx += ss
            t2 = time.time()
            print("time taken to filter", t2 - t1)
            print("final size is", curidx, nchans)
            jds = unix2jd(ctime_arr)
            south_ant = EarthLocation.from_geodetic(
                lat=51.4646065, lon=-68.2352594, height=341.052
            )
            atime = Time(jds, format="jd", scale="utc", location=south_ant)
            lstimes = atime.sidereal_time("mean").value
            lstbins = np.round(lstimes / dH).astype(int)
            bins = np.unique(lstbins)  # to pass to binning function
            nbins = len(bins)  # to pass to binning function
        else:
            files = scio.read_files(new_fnames)
            curidx = 0
            for fnum, file in enumerate(files):
                # print(fnum)
                large_arr[curidx : curidx + file.shape[0], :] = file
                curidx += file.shape[0]
        # print(lstimes, lstbins)
        nanarr = large_arr
        nanmask = large_arr_mask
        result = np.empty((nbins, nchans), dtype="float64")
        t1 = time.time()
        tools.set_nans(nanarr, nanmask)  # set masked values in the data array to NaN
        t2 = time.time()
        print("time taken to mask", t2 - t1)
        t1 = time.time()
        tools.myfunc(nanarr, lstbins, bins, result)  # LST bin the data
        t2 = time.time()
        print("time taken by lstbin", t2 - t1)
        np.savez_compressed(
            f"/scratch/s/sievers/mohanagr/lstbin__{location}_mega_{tag}_test.npz",
            lstavg=result,
            tag=tag,
            bins=bins,
        )
    # fig,ax=plt.subplots(1,2)
    # fig.set_size_inches(10,5)
    # ax[0].imshow(np.log10(result),aspect="auto",vmin=7,vmax=9)
    # ax[1].plot(np.unique(lstbins,return_counts=True)[1])
    # fig.savefig(f"/scratch/s/sievers/mohanagr/lstbin_{time.time()}.jpg")
