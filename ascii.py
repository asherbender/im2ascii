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

if __name__=="__main__":
