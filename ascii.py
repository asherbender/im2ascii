#!/usr/bin/python
import os
import sys
import argparse
import textwrap
import numpy as np
from skimage import io
import skimage.transform
from skimage.color import rgb2gray


# Characters available for ASCII art.
#
#     ~ !"#$%&'()*+,-./0123456789:;<=
#     >?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]
#     ^_`abcdefghijklmnopqrstuvwxyz{|}
#
CHARS = np.array([126, ] + range(32, 62) + range(62, 94) + range(94, 126))

# Hard-coded perceptual intensity for ASCII characters.
CHAR_INTENSITY = np.array([0.286 , 0.    , 0.4   , 0.2372, 0.8726, 0.7619, 0.6703,
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

    return mean_image


def bin_intensity(bins, characters, intensities):

    # Bin character intensities into discrete levels.
    counts, edges = np.histogram(intensities, bins=bins)

    # Remove bins with zero observations.
    t = [(count, edge) for count, edge in zip(counts, edges) if count > 0]
    counts, edges = zip(*t)
    counts = list(counts)
    edges = list(edges)

    # Make the lightest intensity a category of its own (so white backgrounds
    # are not randomised).
    sorted_intensities = np.sort(intensities)
    if counts[0] > 1:
        counts.insert(0, 1)
        counts[1] -= 1
        edges.insert(1, 0.75 * sorted_intensities[1])

    # Determine which bin the intensities belong to.
    #
    # Note: bins without characters were removed. Approximate these values by
    #       assigning pixels to the nearest bin that has characters.
    edges = np.array(edges)
    intensity_bin = np.digitize(intensities, edges)

    # Create list of choices for each bin.
    choices = list()
    for i in range(1, len(counts) + 1):
        choices.append(characters[intensity_bin == i])

    return edges, choices


def grayscale_to_ascii(img, characters=CHARS, intensities=CHAR_INTENSITY,
                       bins=None, boarder=False):

    # Reshape image and intensity vector into arrays.
    rows, cols = img.shape
    pixels = img.flatten().reshape((1, img.size))
    intensities = intensities.reshape((1, intensities.size))

    # Bin pixel into intensity bin.
    if bins:
        edges, choices = bin_intensity(bins, characters, intensities.flatten())

        # Find closest bin.
        edges = edges.reshape((1, len(edges)))
        nearest_bin = np.abs(edges.T - pixels).argmin(axis=0)

        # Iterate through bins.
        ascii_matrix = np.zeros(nearest_bin.shape, dtype=np.int64)
        for i in range(len(choices)):
            idx = nearest_bin == i
            if len(choices[i]) == 1:
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

    # Initialise ASCII art.
    if boarder:
        ascii_art = '+' + cols * '-' + '+\n'
    else:
        ascii_art = ''

    # Convert matrix to string.
    for i in range(rows):
        row = ''.join(unichr(c) for c in ascii_matrix[i, :])
        if boarder:
            row = '|' + row + '|'

        ascii_art += row + '\n'

    if boarder:
        ascii_art += '+' + cols * '-' + '+\n'

    return ascii_art


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    #         Configure command-line options & parsing behaviour
    # -------------------------------------------------------------------------

    man = """Convert image into ASCII art."""
    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class,
                                     description=textwrap.dedent(man))

    # Image argument.
    parser.add_argument('image', type=str, nargs='?', help='Path to image.')

    # Width.
    msg = 'Width of ASCII art in characters.'
    parser.add_argument('-w', '--width', metavar='[int]', help=msg, type=int,
                        default=80)

    # Intensity levels/bins.
    msg = """By default ASCII art is generated by allocating each pixel to the nearest
    character based on the characters' perceptual intensity. If binning is
    enabled, the characters are arranged into a number of perceptual intensity
    bins. ASCII art is then generated by assigning each pixel to an intensity
    bin and drawing a random character from the bin. Levels must be an integer
    between 5 and 50."""
    parser.add_argument('-l', '--levels', metavar='[int]', help=msg, type=int,
                        default=None)

    # Character ratio.
    msg = """Height:width ratio per ASCII character. The ratio must be greater
    than or equal to one (square characters)."""
    parser.add_argument('-r', '--ratio', metavar='[float]', help=msg,
                        type=float, default=2)

    # Image boarder.
    parser.add_argument('-b', '--boarder', action='store_true', default=False,
                        help='Add boarder to ASCII art.')

    # Print character set.
    msg = """Print character set with perceptual intensity and exit. If binning
    has been enabled, each intensity level and its associated characters will
    be printed."""
    parser.add_argument('--list', action='store_true', default=False, help=msg)

    # -------------------------------------------------------------------------
    #                  Get arguments from the command-line
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    # Ensure the number of intensity bins is valid.
    if (args.levels is not None) and ((args.levels < 5) or (args.levels > 50)):
        raise Exception("'--bins' must be greater 5 and less than 50.")

    # Print characters and exit.
    if args.list:
        if args.levels:
            char_set = zip(*bin_intensity(args.levels, CHARS, CHAR_INTENSITY))
        else:
            char_set = sorted(zip(CHAR_INTENSITY, CHARS), key=lambda t: t[0])

        for intensity, character in char_set:
            if isinstance(character, int):
                string = unichr(character)
            else:
                string = ''.join([unichr(c) for c in character])
            print '%1.4f: %s' % (intensity, string)

        sys.exit()

    # Ensure the image file exists.
    if not os.path.isfile(args.image):
        msg = "The image '%s' does NOT exist."
        raise IOError(msg % args.image)

    # If the boarder has been enabled, make the image slighly smaller so that
    # the final output is the specified width.
    if args.boarder:
        args.width -= 2

    # Ensure the width is valid.
    if args.width < 2:
        raise Exception("'--width' must be greater than or equal to two.")

    # Ensure the height:width ratio is valid.
    if args.ratio < 1:
        raise Exception("'--ratio' must be greater than or equal to one.")

    # -------------------------------------------------------------------------
    #                      Convert image into ASCII
    # -------------------------------------------------------------------------
    character_width = 1
    character_height = args.ratio

    # Load image and convert to grey-scale.
    #
    # Note: the 'as_grey' option in io.imread does not appear to work on all
    #       images.
    #
    img = io.imread(args.image)
    if img.shape == (2,):
        img = rgb2gray(img[0])
    else:
        img = rgb2gray(img)

    # Invert grey-scale: zero is white, one is black.
    img = 1. - img

    # Re-scale image width to number of characters specified. Preserve aspect
    # ratio.
    rescaled_img = scale_image(img, rescaled_width=args.width)

    # Perform averaging on blocks in the re-scaled image. This takes into
    # account the fact that ASCII character are rendered non-square.
    mean_image = average_pixels(rescaled_img,
                                char_width=character_width,
                                char_height=character_height)

    # Convert mean intensity image into ASCII art.
    ascii_art = grayscale_to_ascii(mean_image,
                                   characters=CHARS,
                                   intensities=CHAR_INTENSITY,
                                   bins=args.levels,
                                   boarder=args.boarder)

    print ascii_art
