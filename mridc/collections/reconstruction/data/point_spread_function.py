import matplotlib.pyplot as plt
import numpy as np


def point_spread_function(masktype, shape, iterations, center_fraction, acceleration, scale_factor):
    # apply point spread function to find best mask (as least clustered lines as possible)
    mask_func = create_mask_for_mask_type(masktype, [center_fraction], [acceleration])
    lowest_value = 5
    for i in range(iterations):
        # get mask
        best_mask, best_acc = mask_func([1, shape[0], shape[1], 2], scale=scale_factor)
        # fft mask
        mask3 = np.squeeze(best_mask[0])
        # mask3 = best_mask
        a = np.fft.fftshift(np.fft.fft(mask3))
        b = np.abs(a)
        plt.plot(b)
        plt.show()
        # remove center
        center_frac = 0.08
        center_lines = int((len(b) * center_frac) / 2)
        center = np.argmax(b)
        b[center - center_lines: center] = 0
        b[center: center + center_lines] = 0
        plt.plot(b)
        plt.show()
        # get highest value other than center
        c = np.argsort(b)
        c = b[c]
        highest_value = c[-1]
        if highest_value < lowest_value:
            lowest_value = highest_value
            mask = best_mask
            acc = best_acc
    return mask, acc
