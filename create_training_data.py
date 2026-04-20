from utils.process_and_simulate import *

import os
import pandas as pd
import time
from datetime import timedelta

#Parameters

#NOTE: This script can create a fully simulated dataset, a dataset composed exclusively of modified real data, or a dataset composed of both
#Set the percentages below based on the desired composition of the output dataset

#Simulated

#How many simulated true positive bottles?
circular_perc = 0.1
#How many true negative bottles (blank)?
tn_perc = 0.1
#How many true negatives with the wrong marks?
tn_logo_perc = 0.1

#Modified real data

#How many logos added to real Canny bottles?
real_perc = 0.1
#How many true negatives with marks on real bottles?
tn_logo_real_perc = 0.1
#How many true negatives with no marks on real bottles?
tn_real_perc = 0.1

#How many images total?
image_num = 10
#Output image size
image_size = (1024, 1024) #NOTE each 2048 x 2048 image is about 50 kb; an n = 1000 training dataset of that size is therefore about 50 mb
#To what folder should this new training data be added?
train_folder = 'test-data'
#Which logo should be added to the fully simulated data?
logo_to_add = 'figs/Diamond_I.jpg'
#Which logo(s) should be added to the real images?
logo_to_add_real = ['figs/Diamond_I_Canny.jpg']
#True negative logo folder
#Folder containing other logos to add to true negative data
tn_logo_folder = 'true_negatives'
#Real data backgrounds folder
#Folder containing real images to which logos will be added
real_im_folder = 'Empty_bottles'
#Leave True (for simulated data) unless the logo is white on black
invert_logo = True
#If True, 50/50 chance that stippling will be added to a given image
stipple_random = False
#If False, set this value for True (add stipple) or False (don't add stipple)
stipple_else = True
#Max size of dot to remove (>= 1) - 3 is a good place to start. Smaller marks should not go above 3 for 512 images.
max_removal = 2 #Note that this should also scale with image size
#Default is (3, 6) which is appropriate for smaller marks - higher numbers mean that images will be rescaled down by a greater factor
#Bigger logos need smaller numbers (1, 2) e.g., AB
scale_range_min = 3
scale_range_max = 6

#Should images be white on black (False) or black on white (True)
invert_ims = True

#Run
#Shouldn't need to change anything below this line if the above parameters are correctly set

start = time.time()

dataset_id = random.randint(1000000,9999999)
print(f'Creating dataset {dataset_id}')

circ_images = int(image_num*circular_perc)
tn_images = int(image_num*tn_perc)
tn_logo_images = int(image_num*tn_logo_perc)
real_perc_images = int(image_num*real_perc)
tn_logo_real_perc_ims = int(image_num*tn_logo_real_perc)
tn_real_perc_ims = int(image_num*tn_real_perc)

os.mkdir(f'{train_folder}/{dataset_id}')
os.mkdir(f'{train_folder}/{dataset_id}/images')
os.mkdir(f'{train_folder}/{dataset_id}/masks')

summary_notes = []

if circular_perc != 0:
    for i in range(1, circ_images+1):
        if stipple_random == True:
            stipple = random.choice([True, False])
        else:
            stipple = stipple_else
        result = circular_image(logo_to_add, image_size = image_size, stipple = stipple, invert_logo = invert_logo, scale_range_min=scale_range_min, scale_range_max=scale_range_max, max_removal = max_removal)
        image = result[0]
        mask = result[-1]
        output = img_as_ubyte(image)
        if invert_ims == True:
            output = invert(output)
        out_mask = img_as_ubyte(mask)
        imsave(f'{train_folder}/{dataset_id}/images/{dataset_id}_{i}.png', output, check_contrast = False)
        imsave(f'{train_folder}/{dataset_id}/masks/{dataset_id}_{i}.png', out_mask, check_contrast = False)
        bbox = [str(x) for x in result[2]]
        summary_notes.append([f'{dataset_id}_{i}', 'circle', ",".join(bbox)])
        if i % 10 == 0:
            print(f'Saved {i}/{circ_images} circular images and masks')

if tn_perc != 0:
    for i in range(1, tn_images+1):
        if stipple_random == True:
            stipple = random.choice([True, False])
        else:
            stipple = stipple_else
        result = true_negative_blank(image_size = image_size, stipple = stipple, max_removal = max_removal)
        image = result[0]
        mask = result[1]
        output = img_as_ubyte(image)
        if invert_ims == True:
            output = invert(output)
        out_mask = img_as_ubyte(mask)
        imsave(f'{train_folder}/{dataset_id}/images/{dataset_id}_{circ_images + i}.png', output, check_contrast = False)
        imsave(f'{train_folder}/{dataset_id}/masks/{dataset_id}_{circ_images + i}.png', out_mask, check_contrast = False)
        bbox = 'No bounding box'
        summary_notes.append([f'{dataset_id}_{circ_images + i}', 'true negative', bbox])
        if i % 10 == 0:
            print(f'Saved {i}/{tn_images} true negative images and masks')

if tn_logo_perc != 0:
    for i in range(1, tn_logo_images+1):
        if stipple_random == True:
            stipple = random.choice([True, False])
        else:
            stipple = stipple_else
        image, mask, logoname = true_negative_logo(tn_logo_folder, image_size = image_size, stipple = stipple, max_removal = max_removal, scale_range_min = scale_range_min, scale_range_max = scale_range_min)
        logo_id = logoname.split('.')[0]
        output = img_as_ubyte(image)
        if invert_ims == True:
            output = invert(output)
        out_mask = img_as_ubyte(mask)
        imsave(f'{train_folder}/{dataset_id}/images/{dataset_id}_{circ_images + tn_images + i}.png', output, check_contrast = False)
        imsave(f'{train_folder}/{dataset_id}/masks/{dataset_id}_{circ_images + tn_images + i}.png', out_mask, check_contrast = False)
        bbox = 'No bounding box'
        summary_notes.append([f'{dataset_id}_{circ_images + tn_images + i}', f'true negative ({logo_id})', bbox])
        if i % 10 == 0:
            print(f'Saved {i}/{tn_logo_images} true negative logo images and masks')

if real_perc != 0:
    for i in range(1, real_perc_images+1):
        logo_choice = random.choice(logo_to_add_real)
        real_im = random.choice(os.listdir(real_im_folder))
        image, mask = modify_real_data(f'{real_im_folder}/{real_im}', logo_choice, image_size = image_size)
        if image.dtype == 'bool':
            output = img_as_ubyte(image)
        else:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            output = img_as_ubyte(image)
        if invert_ims == False:
            output = invert(output)
        out_mask = img_as_ubyte(mask)
        imsave(f'{train_folder}/{dataset_id}/images/{dataset_id}_{circ_images + tn_images + tn_logo_images + i}.png', output, check_contrast = False)
        imsave(f'{train_folder}/{dataset_id}/masks/{dataset_id}_{circ_images + tn_images + tn_logo_images + i}.png', out_mask, check_contrast = False)
        summary_notes.append([f'{dataset_id}_{circ_images + tn_images + tn_logo_images + i}', 'true positive (added to real data)', ''])
        if i % 10 == 0:
            print(f'Saved {i}/{real_perc_images} "real" images with {logo_to_add_real}')

if tn_logo_real_perc != 0:
    for i in range(1, tn_logo_real_perc_ims+1):
        real_im = random.choice(os.listdir(real_im_folder))
        fake_logo = random.choice(os.listdir(tn_logo_folder))
        image, mask = modify_real_data(f'{real_im_folder}/{real_im}', f'{tn_logo_folder}/{fake_logo}', TN = True, image_size = image_size, add_canny = True, add_mask = False)
        if image.dtype == 'bool':
            output = img_as_ubyte(image)
        else:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            output = img_as_ubyte(image)
        # output = img_as_ubyte(image)
        if invert_ims == False:
            output = invert(output)
        out_mask = img_as_ubyte(mask)
        imsave(f'{train_folder}/{dataset_id}/images/{dataset_id}_{circ_images + + tn_images + tn_logo_images + real_perc_images + i}.png', output, check_contrast = False)
        imsave(f'{train_folder}/{dataset_id}/masks/{dataset_id}_{circ_images + + tn_images + tn_logo_images + real_perc_images + i}.png', out_mask, check_contrast = False)
        summary_notes.append([f'{dataset_id}_{circ_images + + tn_images + tn_logo_images + real_perc_images + i}', 'true negative (logo added to real data)' ,''])
        if i % 10 == 0:
            print(f'Saved {i}/{tn_logo_real_perc_ims} "real" true negative images with fake logos')

if tn_real_perc_ims != 0:
    for i in range(1, tn_real_perc_ims + 1):
        real_im = random.choice(os.listdir(real_im_folder))
        image, mask = modify_real_data(f'{real_im_folder}/{real_im}', logo_to_add= None, TN = True, image_size = image_size, add_canny = True, add_mask = False)
        if image.dtype == 'bool':
            output = img_as_ubyte(image)
        else:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            output = img_as_ubyte(image)
        if invert_ims == False:
            output = invert(output)
        out_mask = img_as_ubyte(mask)
        id_stop = circ_images + tn_images + tn_logo_images + real_perc_images + tn_logo_real_perc_ims
        # id_stop = 2250
        imsave(f'{train_folder}/{dataset_id}/images/{dataset_id}_{id_stop + i}.png', output, check_contrast = False)
        imsave(f'{train_folder}/{dataset_id}/masks/{dataset_id}_{id_stop + i}.png', out_mask, check_contrast = False)
        summary_notes.append([f'{dataset_id}_{circ_images + tn_images + tn_logo_images + real_perc_images + tn_logo_real_perc_ims + i}', 'true negative (no logo, real data)' ,''])
        if i % 10 == 0:
            print(f'Saved {i}/{tn_real_perc_ims} "real" true negative images')

df = pd.DataFrame(summary_notes, columns = ['image_id', 'image_type', 'bbox coords'])
df.to_csv(f'{train_folder}/{dataset_id}/{dataset_id}_summary.csv')

end = time.time()

print(f'{str(timedelta(seconds = end - start))} elapsed')