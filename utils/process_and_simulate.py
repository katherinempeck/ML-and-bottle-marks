import math

import numpy as np

import os

import random

from skimage import img_as_float, exposure
from skimage.color import rgb2gray
from skimage.draw import disk as drawdisk
from skimage.draw import circle_perimeter, bezier_curve, random_shapes
from skimage.feature import canny
from skimage.filters import gaussian, threshold_otsu
from skimage.io import imread, imsave
from skimage.measure import regionprops, label
from skimage.morphology import dilation, disk, remove_small_holes, convex_hull_image
from skimage.transform import AffineTransform, rescale, resize, rotate, warp
from skimage.util import invert, random_noise, img_as_ubyte

# Process

def process_image(in_image, out_image):
    """
    Open, process, and save in_image as out_image
    """
    image = imread(in_image, as_gray = True)
    image = img_as_float(image)
    output = exposure.equalize_adapthist(image)
    output = canny(output, sigma = 3)
    imsave(out_image, output)

# Simulated

## Fully simulated

def circular_image(logo:str, image_size = (512, 512), shape_buffer = 5, stipple = True, invert_logo = True, scale_range_min = 3, scale_range_max = 6, max_removal = 3):
    """
    Generates a Canny edge detector-filtered image simulating a circular object with the chosen logo applied

    Parameters
    ----------
    logo: str
        file path to two-color image (any format that scikit-image can read) of logo to apply to simulated object
    image_size: tuple
        size of output image, (512, 512) by default
    shape_buffer: integer
        how much dilation should by applied to the edges of the simulated object (5 by default)
    stipple: bool
        if True, add randomly placed dots (radius = 3) to image to stimulate stippling
    invert_logo: bool
        if True, invert the imported logo image (if the image is black on white, it will add 0s - but we want it to add 1s to indicate features)

    Returns
    -------
    image: np.array
        Canny-filtered image, as array (can be saved with imsave())
    final_mask: np.array
        Binary array the same size as image, with logo area represented by 1s (calculated from segmentation mask)
    bbox_coords: list
        List of coordinates for logo bounding box in order minimum y (i.e., min. row), minimum x, maximum y, maximum x
    detailed_mask: np.array
        Array holding segmentation mask

    """
    #Import logo image and convert to grayscale
    logo = imread(logo)
    if len(logo.shape) == 3:
        logo = rgb2gray(logo)
    else:
        pass
    threshold = threshold_otsu(logo)
    logo = logo > threshold
    #Add noise to logo only
    logo = random_noise(logo, mode = 'speckle', clip = True, mean = random.uniform(-.1, -.2))
    length, width = logo.shape
    #Create blank image at specified size; (512, 512) is the default
    image = np.zeros(image_size)
    #Create parameters for randomly sized circle
    #This will represent the exterior of the object (bottle-like) to which the logo is added
    #Select a random percentage - this will determine how much smaller the exterior circle is than the image itself
    size_modifier = random.uniform(0.8, 0.95)
    #Get center coordinates of image
    center_x = int(math.floor(image.shape[0]/2))
    center_y = int(math.floor(image.shape[1]/2))
    #Determine size of circle by multiplying the size modified by the image size
    size = int(math.floor(image.shape[1]/2)*size_modifier)
    #Resize logo so that it could fit within the circle
    rval = random.uniform(scale_range_min, scale_range_max)
    logo = rescale(logo, center_x/length/rval)
    if invert_logo == True:
        logo = invert(logo)
    #Add circle to image
    rr, cc = circle_perimeter(center_x, center_y, size)
    image[rr, cc] = 1
    #Buffer shape
    image = dilation(image, disk(shape_buffer))
    #Add randomly buffered circles around the edges of the shape to create more noise
    for i in range(1, random.randrange(1, 10)):
        factor = round(random.uniform(-20, 20))
        new_size = size + factor
        if new_size*2 >= image.shape[1]:
            pass
        else:
            yy, xx = circle_perimeter(center_x, center_y, new_size)
            image[yy, xx] = 1
    #Buffer added and original circles slightly
    image = dilation(image, disk(1))
    #Blur before adding logo to add more noise to background
    image = gaussian(image, sigma = 1)
    #Place logo in shape
    #https://stackoverflow.com/questions/58248121/opencv-python-how-to-overlay-an-image-into-the-centre-of-another-image
    h, w = logo.shape
    hh, ww = image.shape
    yoff = round((hh-h)/2+ random.uniform(10, 50))
    xoff = round((ww-w)/2)
    image[yoff:yoff+h, xoff:xoff+w] = logo
    threshold = threshold_otsu(image)
    image = image > threshold
    #Create image that is just a mask of the logo
    detailed_mask = np.zeros(image_size)
    detailed_mask[yoff:yoff+h, xoff:xoff+w] = logo
    threshold = threshold_otsu(detailed_mask)
    detailed_mask = detailed_mask > threshold
    #Fill holes in mask
    detailed_mask = remove_small_holes(detailed_mask, max_size = max(yoff+h, xoff+w)*max(yoff+h, xoff+w))
    #Add stippling
    if stipple == True:
        rrd, ccd = drawdisk((center_x, center_y), size)
        coords = [(int(x), int(y)) for x, y in zip(ccd, rrd)]
        coord_sample = random.sample(coords, size*2)
        for s in coord_sample:
            yy, xx = drawdisk(s, 3)
            image[xx, yy] = 1
    #Rotate image (and mask) randomly
    rotation_val = random.uniform(0, 180)
    image = rotate(image, rotation_val)
    detailed_mask = rotate(detailed_mask, rotation_val)
    #Skew the image and mask using an affine transformation
    skew_val = np.pi/random.uniform(20, 30)
    tform = AffineTransform(shear = skew_val)
    skew_type = random.choice([tform, tform.inverse])
    image = warp(image, skew_type)
    detailed_mask = warp(detailed_mask, skew_type)
    #Add random Bezier curves
    #These are curved lines used to add additional noise on/around the logo
    #Simulate noise that could come from photo backdrops or fingers holding an object
    for i in range(1, random.randrange(1, 20)):
        r0 = random.randrange(0, image.shape[0])
        c0 = random.randrange(0, image.shape[1])
        r1 = random.randrange(0, image.shape[0])
        r2 = random.randrange(0, image.shape[0])
        c1 = random.randrange(0, image.shape[1])
        c2 = random.randrange(0, image.shape[1])
        brr, bcc = bezier_curve(r0, c0, r1, c1, r2, c2, weight = random.uniform(1, 10), shape = image.shape)
        image[brr, bcc] = 1
    #Blur
    image = gaussian(image, sigma = 3)
    #Noise
    image = random_noise(image, mode = 'speckle', clip = True, mean = random.uniform(-.1, -.2))
    #Canny
    image = canny(image, sigma = 1.5)
    #Remove chunks from image by randomly generating circles with value 0 inside bbox
    #NOTE might require some fine-tuning to make sure the logo is not being fully removed
    labeled = label(detailed_mask)
    min_row, min_col, max_row, max_col = regionprops(labeled)[0].bbox
    final_mask = np.zeros(image_size)
    final_mask[min_row:max_row, min_col:max_col] = 1
    bbox_coords = [min_row, min_col, max_row, max_col]
    rrch, ccch = np.where(detailed_mask == 1)
    coords = [(int(x), int(y)) for x, y in zip(ccch, rrch)]
    coord_sample = random.sample(coords, int(round((min(xoff+w, yoff+h))/random.uniform(1.1, 2))))
    for s in coord_sample:
        if stipple == True:
            yy, xx = drawdisk(s, random.uniform(1, max_removal/2))
        else:
            yy, xx = drawdisk(s, random.uniform(1, max_removal))
        image[xx, yy] = 0
    #Add Perlin noise
    image = img_as_ubyte(image)
    out = generate_perlin_noise_2d(image_size, (64, 64))
    out = canny(out)
    out = invert(out)
    out = img_as_ubyte(out)
    shp, _ = random_shapes(image_size, min_shapes=8, max_shapes=40, min_size = 10, max_size= 100, allow_overlap=True, num_channels = 1)
    shp = shp[:, :, 0]
    threshold = threshold_otsu(shp)
    shp = shp > threshold
    m = np.ma.masked_where(shp == True, shp)
    new_x = np.ma.masked_where(np.ma.getmask(m), out).filled(255)
    new_x = (new_x - np.min(new_x)) / (np.max(new_x) - np.min(new_x))
    new_x = invert(new_x)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image + new_x
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image > 0
    image = img_as_ubyte(image)
    return image, final_mask, bbox_coords, detailed_mask

def true_negative_blank(image_size = (512, 512), stipple = True, max_removal = 3, shape_buffer = 5):
    #Create parameters for randomly sized circle
    #This will represent the exterior of the object (bottle-like) to which the logo is added
    #Select a random percentage - this will determine how much smaller the exterior circle is than the image itself
    size_modifier = random.uniform(0.8, 0.95)
    image = np.zeros(image_size)
    #Get center coordinates of image
    center_x = int(math.floor(image.shape[0]/2))
    center_y = int(math.floor(image.shape[1]/2))
    #Determine size of circle by multiplying the size modified by the image size
    size = int(math.floor(image.shape[1]/2)*size_modifier)
    #Add circle to image
    rr, cc = circle_perimeter(center_x, center_y, size)
    image[rr, cc] = 1
    #Buffer shape
    image = dilation(image, disk(shape_buffer))
    #Add randomly buffered circles around the edges of the shape to create more noise
    for i in range(1, random.randrange(1, 10)):
        factor = round(random.uniform(-20, 20))
        new_size = size + factor
        if new_size*2 >= image.shape[1]:
            pass
        else:
            yy, xx = circle_perimeter(center_x, center_y, new_size)
            image[yy, xx] = 1
    #Buffer added and original circles slightly
    image = dilation(image, disk(1))
    #Blur before adding logo to add more noise to background
    image = gaussian(image, sigma = 1)
    #Add stippling
    if stipple == True:
        rrd, ccd = drawdisk((center_x, center_y), size)
        coords = [(int(x), int(y)) for x, y in zip(ccd, rrd)]
        coord_sample = random.sample(coords, size*2)
        for s in coord_sample:
            yy, xx = drawdisk(s, 3)
            image[xx, yy] = 1
    #Rotate image (and mask) randomly
    rotation_val = random.uniform(0, 180)
    image = rotate(image, rotation_val)
    #Skew the image and mask using an affine transformation
    skew_val = np.pi/random.uniform(20, 30)
    tform = AffineTransform(shear = skew_val)
    skew_type = random.choice([tform, tform.inverse])
    image = warp(image, skew_type)
    for i in range(1, random.randrange(1, 20)):
        r0 = random.randrange(0, image.shape[0])
        c0 = random.randrange(0, image.shape[1])
        r1 = random.randrange(0, image.shape[0])
        r2 = random.randrange(0, image.shape[0])
        c1 = random.randrange(0, image.shape[1])
        c2 = random.randrange(0, image.shape[1])
        brr, bcc = bezier_curve(r0, c0, r1, c1, r2, c2, weight = random.uniform(1, 10), shape = image.shape)
        image[brr, bcc] = 1
    #Blur
    image = gaussian(image, sigma = 3)
    #Noise
    image = random_noise(image, mode = 'speckle', clip = True, mean = random.uniform(-.1, -.2))
    #Canny
    image = canny(image, sigma = 1.5)
    image = image > 0
    if stipple == False:
        rrd, ccd = drawdisk((center_x, center_y), size)
        coords = [(int(x), int(y)) for x, y in zip(ccd, rrd)]
        coord_sample = random.sample(coords, size*2)
    for s in coord_sample:
        if stipple == True:
            yy, xx = drawdisk(s, random.uniform(1, 2))
        else:
            yy, xx = drawdisk(s, random.uniform(1, max_removal))
        image[xx, yy] = 0
        #Add Perlin noise
    image = img_as_ubyte(image)
    out = generate_perlin_noise_2d(image_size, (64, 64))
    out = canny(out)
    out = invert(out)
    out = img_as_ubyte(out)
    shp, _ = random_shapes(image_size, min_shapes=8, max_shapes=40, min_size = 10, max_size= 100, allow_overlap=True, num_channels = 1)
    shp = shp[:, :, 0]
    threshold = threshold_otsu(shp)
    shp = shp > threshold
    m = np.ma.masked_where(shp == True, shp)
    new_x = np.ma.masked_where(np.ma.getmask(m), out).filled(255)
    new_x = (new_x - np.min(new_x)) / (np.max(new_x) - np.min(new_x))
    new_x = invert(new_x)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image + new_x
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image > 0
    image = img_as_ubyte(image)
    mask = np.zeros(image_size)
    return image, mask

def true_negative_logo(logo_folder, image_size = (512, 512), shape_buffer = 5, stipple = True, invert_logo = True, scale_range_min = 3, scale_range_max = 6, max_removal = 3):
    #Import logo image and convert to grayscale
    logo_list = [f'{logo_folder}/{l}' for l in os.listdir(logo_folder)]
    logo_choice = random.choice(logo_list)
    logo = imread(logo_choice)
    if len(logo.shape) == 3:
        logo = rgb2gray(logo)
    else:
        pass
    #Add noise to logo only
    logo = random_noise(logo, mode = 'speckle', clip = True, mean = random.uniform(-.1, -.2))
    length, width = logo.shape
    #Create blank image at specified size (512, 512) is the default
    image = np.zeros(image_size)
    #Create parameters for randomly sized circle
    #This will represent the exterior of the object (bottle-like) to which the logo is added
    #Select a random percentage - this will determine how much smaller the exterior circle is than the image itself
    size_modifier = random.uniform(0.8, 0.95)
    #Get center coordinates of image
    center_x = int(math.floor(image.shape[0]/2))
    center_y = int(math.floor(image.shape[1]/2))
    #Determine size of circle by multiplying the size modified by the image size
    size = int(math.floor(image.shape[1]/2)*size_modifier)
    #Resize logo so that it could fit within the circle
    #TODO rescale logo to be a bit smaller
    rval = random.uniform(scale_range_min, scale_range_max)
    logo = rescale(logo, center_x/length/rval)
    if invert_logo == True:
        logo = invert(logo)
    #Add circle to image
    rr, cc = circle_perimeter(center_x, center_y, size)
    image[rr, cc] = 1
    #Buffer shape
    image = dilation(image, disk(shape_buffer))
    #Add randomly buffered circles around the edges of the shape to create more noise
    for i in range(1, random.randrange(1, 10)):
        factor = round(random.uniform(-20, 20))
        new_size = size + factor
        if new_size*2 >= image.shape[1]:
            pass
        else:
            yy, xx = circle_perimeter(center_x, center_y, new_size)
            image[yy, xx] = 1
    #Buffer added and original circles slightly
    image = dilation(image, disk(1))
    #Blur before adding logo to add more noise to background
    image = gaussian(image, sigma = 1)
    #Place logo in shape
    #https://stackoverflow.com/questions/58248121/opencv-python-how-to-overlay-an-image-into-the-centre-of-another-image
    h, w = logo.shape
    hh, ww = image.shape
    yoff = round((hh-h)/2+ random.uniform(10, 50))
    xoff = round((ww-w)/2)
    image[yoff:yoff+h, xoff:xoff+w] = logo
    #Add stippling
    if stipple == True:
        rrd, ccd = drawdisk((center_x, center_y), size)
        coords = [(int(x), int(y)) for x, y in zip(ccd, rrd)]
        coord_sample = random.sample(coords, size*2)
        for s in coord_sample:
            yy, xx = drawdisk(s, 3)
            image[xx, yy] = 1
    #Rotate image randomly
    rotation_val = random.uniform(0, 180)
    image = rotate(image, rotation_val)
    #Skew the image and mask using an affine transformation
    skew_val = np.pi/random.uniform(20, 30)
    tform = AffineTransform(shear = skew_val)
    skew_type = random.choice([tform, tform.inverse])
    image = warp(image, skew_type)
    #Add random Bezier curves
    #These are curved lines used to add additional noise on/around the logo
    #Simulate noise that could come from photo backdrops or fingers holding an object
    for i in range(1, random.randrange(1, 20)):
        r0 = random.randrange(0, image.shape[0])
        c0 = random.randrange(0, image.shape[1])
        r1 = random.randrange(0, image.shape[0])
        r2 = random.randrange(0, image.shape[0])
        c1 = random.randrange(0, image.shape[1])
        c2 = random.randrange(0, image.shape[1])
        brr, bcc = bezier_curve(r0, c0, r1, c1, r2, c2, weight = random.uniform(1, 10), shape = image.shape)
        image[brr, bcc] = 1
    #Blur
    image = gaussian(image, sigma = 3)
    #Noise
    image = random_noise(image, mode = 'speckle', clip = True, mean = random.uniform(-.1, -.2))
    #Canny
    image = canny(image, sigma = 1.5)
    image = image > 0
    #Remove chunks from image by randomly generating circles with value 0 inside bbox
    #NOTE might require some fine-tuning to make sure the logo is not being fully removed
    if stipple == False:
        rrd, ccd = drawdisk((center_x, center_y), size)
        coords = [(int(x), int(y)) for x, y in zip(ccd, rrd)]
    coord_sample = random.sample(coords, int(round((min(xoff+w, yoff+h))/random.uniform(1.1, 2))))
    for s in coord_sample:
        if stipple == True:
            yy, xx = drawdisk(s, random.uniform(1,  max_removal/2))
        else:
            yy, xx = drawdisk(s, random.uniform(1, max_removal))
        image[xx, yy] = 0
        #Add Perlin noise
    image = img_as_ubyte(image)
    out = generate_perlin_noise_2d(image_size, (64, 64))
    out = canny(out)
    out = invert(out)
    out = img_as_ubyte(out)
    shp, _ = random_shapes(image_size, min_shapes=8, max_shapes=40, min_size = 10, max_size= 100, allow_overlap=True, num_channels = 1)
    shp = shp[:, :, 0]
    threshold = threshold_otsu(shp)
    shp = shp > threshold
    m = np.ma.masked_where(shp == True, shp)
    new_x = np.ma.masked_where(np.ma.getmask(m), out).filled(255)
    new_x = (new_x - np.min(new_x)) / (np.max(new_x) - np.min(new_x))
    new_x = invert(new_x)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image + new_x
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image > 0
    image = img_as_ubyte(image)
    #Create blank mask for true negative
    mask = np.zeros(image_size)
    return image, mask, os.path.basename(logo_choice)

## Modified real data

#Most of the real Canny images have small, noisy areas with sinuous, interlocking, maze-like lines
#These are difficult to replicate simply with random Bezier curves
#Adding small circles does help, but they are generally too dispersed to look like the noise in the images
#Perlin noise looks much closer to the actual noise in the CAnny images

#The code below this line is via https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

# End code from https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html

def modify_real_data(real_canny, logo_to_add = None, TN = False, image_size = (512, 512), invert_logo = False, scale_range_min = 3, scale_range_max = 6, add_canny = False, add_mask = True, max_removal = 3):
    #Open "real" image as background
    bg = imread(real_canny, as_gray = True)
    # bg = rgb2gray(bg)
    bg_resized = resize(
        bg, image_size, anti_aliasing=True
    )
    if logo_to_add is not None:
        #Prepare logo
        logo = imread(logo_to_add)
        if len(logo.shape) == 3:
            logo = rgb2gray(logo)
        else:
            pass
        #Augment logo
        if random.randint(0, 100) < 50:
            logo = np.flipud(logo)
        if random.randint(0, 100) < 50:
            logo = np.fliplr(logo)
        #Resize logo so that it could fit within the circle
        rsc = random.randrange(scale_range_min, scale_range_max)
        logo = resize(logo, (int(logo.shape[0]/rsc), int(logo.shape[1]/rsc)))
        logo = rotate(logo, 90, resize = True)
        # logo = resize(logo, ())
        threshold = threshold_otsu(logo)
        logo = logo > threshold
        if invert_logo == True:
            logo = invert(logo)
        if add_canny == True:
            logo = canny(logo)
            logo = invert(logo)
        #Place logo in shape
        #https://stackoverflow.com/questions/58248121/opencv-python-how-to-overlay-an-image-into-the-centre-of-another-image
        h, w = logo.shape
        hh, ww = bg_resized.shape
        yoff = round((hh-h)/2+ random.uniform(10, 50))
        xoff = round((ww-w)/2)
        bg_resized[yoff:yoff+h, xoff:xoff+w] = logo
        threshold = threshold_otsu(bg_resized)
        image = bg_resized > threshold
        #Generate mask
        #Create image that is just a mask of the logo
        detailed_mask = np.ones(image_size)
        detailed_mask[yoff:yoff+h, xoff:xoff+w] = logo
        detailed_mask = invert(detailed_mask)
        #Fill it in
        threshold = threshold_otsu(detailed_mask)
        detailed_mask = detailed_mask > threshold
        detailed_mask = convex_hull_image(detailed_mask)
    else:
        threshold = threshold_otsu(bg_resized)
        image = bg_resized > threshold
        detailed_mask = np.zeros(image_size)
    #Rotate image (and mask) randomly
    rotation_val = random.uniform(-180, 180)
    image = rotate(image, rotation_val, mode = 'constant', cval = 1)
    detailed_mask = rotate(detailed_mask, rotation_val)
    #Skew the image and mask using an affine transformation
    skew_val = np.pi/random.uniform(20, 30)
    tform = AffineTransform(shear = skew_val)
    skew_type = random.choice([tform, tform.inverse])
    image = warp(image, skew_type, mode = 'constant', cval = 1)
    detailed_mask = warp(detailed_mask, skew_type)
    #Random flip
    #Vertical
    if random.randint(0, 100) < 50:
        image = np.flipud(image)
        detailed_mask = np.flipud(detailed_mask)
    #Horizontal
    if random.randint(0, 100) < 50:
        image = np.fliplr(image)
        detailed_mask = np.fliplr(detailed_mask)
    #Random noise
    if random.randint(0, 100) < 25:
        image = random_noise(image, mode = 'salt')
    if logo_to_add is not None:
        rrch, ccch = np.where(detailed_mask == 1)
        coords = [(int(x), int(y)) for x, y in zip(ccch, rrch)]
        samplenum = int(round((min(xoff+w, yoff+h))/random.uniform(1.5, 2.5)))
        coord_sample = random.sample(coords, samplenum)
        for s in coord_sample:
            yy, xx = drawdisk(s, random.uniform(1, max_removal))
            image[xx, yy] = 1
        rrch, ccch = np.where(detailed_mask == 1)
    #Stipple
    rrch, ccch = np.where(image[2:-2, 2:-2] == 1)
    coords = [(int(x), int(y)) for x, y in zip(ccch, rrch)]
    coord_sample = random.sample(coords, 150)
    for s in coord_sample:
        yy, xx = circle_perimeter(s[0], s[1], int(round(random.uniform(1, max_removal))))
        image[xx, yy] = 0
    #Add Perlin noise
    image = img_as_ubyte(image)
    out = generate_perlin_noise_2d(image_size, (64, 64))
    out = canny(out)
    out = invert(out)
    out = img_as_ubyte(out)
    shp, _ = random_shapes(image_size, min_shapes=8, max_shapes=40, min_size = 10, max_size= 100, allow_overlap=True, num_channels = 1)
    shp = shp[:, :, 0]
    threshold = threshold_otsu(shp)
    shp = shp > threshold
    m = np.ma.masked_where(shp == True, shp)
    new_x = np.ma.masked_where(np.ma.getmask(m), out).filled(255)
    new_x = (new_x - np.min(new_x)) / (np.max(new_x) - np.min(new_x))
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image + new_x
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = img_as_ubyte(image)
    if TN == True:
        detailed_mask = np.zeros(image_size)
    return image, detailed_mask