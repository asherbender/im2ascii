#!/usr/bin/python
import numpy as np
from skimage import io
import skimage.transform

# Characters available for ASCII art.
#
#     ~ !"#$%&'()*+,-./0123456789:;<=
#     >?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]
#     ^_`abcdefghijklmnopqrstuvwxyz{|}
#
CHARS = np.array([126, ] + range(32, 62) + range(62, 94) + range(94, 126))

# Hard-coded intensity for ASCII characters.
CHAR_INTENSITY = array([0.286 , 0.    , 0.4   , 0.2372, 0.8726, 0.7619, 0.6703,
                        0.8285, 0.1182, 0.4071, 0.4082, 0.3452, 0.4034, 0.1594,
                        0.105 , 0.0943, 0.393 , 0.8866, 0.5729, 0.6595, 0.6895,
                        0.715 , 0.7202, 0.8348, 0.5213, 0.9029, 0.8347, 0.1881,
                        0.2531, 0.4249, 0.4432, 0.425 , 0.4503, 1.    , 0.7863,
                        0.9914, 0.5933, 0.8796, 0.7962, 0.6399, 0.7805, 0.8587,
                        0.633 , 0.5692, 0.8326, 0.5232, 0.9939, 0.99  , 0.8496,
                        0.7567, 0.9118, 0.9028, 0.7209, 0.5663, 0.7936, 0.6806,
                        0.9691, 0.7229, 0.5608, 0.7121, 0.4963, 0.393 , 0.4963,
                        0.2565, 0.1201, 0.0817, 0.6916, 0.7858, 0.4655, 0.7853,
                        0.6732, 0.5228, 0.866 , 0.6904, 0.482 , 0.5152, 0.6896,
                        0.4605, 0.7789, 0.5899, 0.642 , 0.7846, 0.7862, 0.3867,
                        0.5409, 0.5098, 0.5893, 0.4972, 0.6749, 0.5276, 0.6201,
                        0.5132, 0.5521, 0.4324, 0.5459])


def scale_image(img, rescaled_width=80):

    # Get size of image.
    height, width = img.shape

    # Calculate height of re-scaled image in pixels (preserving aspect ratio).
    rescaled_height = int(float(height * rescaled_width) / float(width))

    # Re-scale image.
    return skimage.transform.resize(img,
                                    (rescaled_height, rescaled_width),
                                    clip=True)


def average_pixels(img, char_width=1, char_height=2, normalise=True):

    # Get size of image.
    height, width = img.shape

    # Calculate number of blocks, in rows and columns, to perform averaging.
    rows = int(float(height) / float(char_height))
    cols = int(float(width) / float(char_width))

    # Average image in blocks.
    mean_image = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mean_image[i, j] = img[(i*char_height):((i+1)*char_height),
                                   (j*char_width):((j+1)*char_width)].mean()

    # Normalise image to be between 0 and 1.
    if normalise:
        mean_image -= mean_image.min()
        mean_image /= mean_image.max()

    return mean_image, rows, cols


def grayscale_to_ascii(img, characters=CHARS, intensities=CHAR_INTENSITY,
                       bins=None):

    # Reshape image and intensity vector into arrays.
    rows, cols = img.shape
    pixels = img.flatten().reshape((1, img.size))
    intensities = intensities.reshape((1, intensities.size))

    # Bin pixel into intensity bin.
    if bins:
        counts, edges = np.histogram(intensities, bins=bins)

        # Remove bins with zero observations.
        t = [(count, edge) for count, edge in zip(counts, edges) if count > 0]
        counts, edges = zip(*t)
        counts = list(counts)
        edges = list(edges)

        # If the white space character exists and has been binned with other
        # characters, make white space a category of its own.
        sorted_intensities = np.sort(intensities.flatten())
        if sorted_intensities[0] == 0 and counts[0] > 1:
            counts.insert(0, 1)
            counts[1] -= 1
            edges.insert(1, 0.75 * sorted_intensities[1])

        # Determine which bin the intensities belong to.
        edges = np.array(edges)
        intensity_bin = np.digitize(intensities.flatten(), edges)

        # Create list of choices for each bin.
        choices = list()
        for i in range(1, len(counts) + 1):
            choices.append(characters[intensity_bin == i])

        # Find closest bin.
        edges = edges.reshape((1, len(edges)))
        nearest_bin = np.abs(edges.T - pixels).argmin(axis=0)

        # Iterate through bins.
        ascii_matrix = np.zeros(nearest_bin.shape, dtype=np.int64)
        for i in range(len(counts)):
            idx = nearest_bin == i
            if counts[i] == 1:
                ascii_matrix[idx] = choices[i]
            else:
                ascii_matrix[idx] = np.random.choice(choices[i],
                                                     size=idx.sum(),
                                                     replace=True)

        # Reshape matrix back to image dimensions.
        ascii_matrix = ascii_matrix.reshape((rows, cols))

    # Bin pixels into nearest registered intensity.
    else:
        idx = np.abs(intensities.T - pixels).argmin(axis=0).reshape((rows, cols))

        # Convert intensity to ASCII characters.
        ascii_matrix = characters[idx].reshape((rows, cols))

    # Convert matrix to string.
    ascii_art = ''
    for i in range(rows):
        ascii_art += ''.join(unichr(c) for c in ascii_matrix[i, :]) + '\n'

    return ascii_art


if __name__ == "__main__":
    pass
