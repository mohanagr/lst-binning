import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation
import warnings
# warnings.filterwarnings("error")
def get_rfi_occupancy(data, deg=4, thresh=5, filter_size=5, get_flags=True):
    predval = np.zeros(data.shape, dtype="float64")
    polyfits = np.polyfit(np.arange(data.shape[0]), np.log10(data), deg)
    # remove time variation using polynomials
    # instead of a simple median subtraction
    for col in range(data.shape[1]):  # fit 2048 polynomials (one for each chan)
        predval[:, col] = np.polyval(polyfits[:, col], np.arange(data.shape[0]))
    postsub3 = np.log10(data) - predval
    # median filter each row (along frequency axis to estimate
    # systematic frequency behaviour
    medfilt3 = median_filter(postsub3, size=filter_size, mode="nearest", axes=1)
    final = (
        postsub3 - medfilt3
    )  # subtract estimated bandpass for each row from the corresponding row
    finalmad = median_abs_deviation(final.flatten())
    wherehigh = np.abs(final) > (thresh * finalmad)
    if get_flags:
        return wherehigh
    rfiocc = np.sum(wherehigh, axis=0) / data.shape[0]
    return rfiocc
