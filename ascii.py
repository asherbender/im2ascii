#!/usr/bin/python

def scale_image(img, rescaled_width=80):

    # Get size of image.
    height, width = img.shape

    # Calculate height of re-scaled image in pixels (preserving aspect ratio).
    rescaled_height = int(float(height * rescaled_width) / float(width))

    # Re-scale image.
    return skimage.transform.resize(img,
                                    (rescaled_height, rescaled_width),
                                    clip=True)


def average_pixels(img, char_width=1, char_height=2):

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

    return mean_image, rows, cols


def grayscale_to_ascii(img, characters=CHARS, intensities=CHAR_INTENSITY):

    # Reshape image and intensity vector into arrays.
    rows, cols = img.shape
    pixels = img.flatten().reshape((1, img.size))
    intensities = intensities.reshape((1, intensities.size))

    # Bin pixels into nearest registered intensity.
    idx = np.abs(intensities.T - pixels).argmin(axis=0).reshape((rows, cols))

    # Convert intensity to ASCII characters.
    ascii_art = ''
    ascii_matrix = characters[idx].reshape((rows, cols))
    for i in range(rows):
        ascii_art += ''.join(unichr(c) for c in ascii_matrix[i, :]) + '\n'

    return ascii_art


if __name__=="__main__":
