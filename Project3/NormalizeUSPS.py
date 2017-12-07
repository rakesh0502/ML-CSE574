import os
from PIL import Image

for image_file_name in os.listdir('USPS_data/Test/'):
    if image_file_name.endswith(".png"):

        im = Image.open('USPS_data/Test/'+image_file_name)
        new_width = 28
        new_height = 28
        im = im.convert('1');
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        im.save('USPS_norm_data/Test/' + image_file_name.replace('.png', '') + '_norm'+'.png')

for i in range(0,10):
    for image_file_name in os.listdir('USPS_data/Numerals/'+str(i)+'/'):
        if image_file_name.endswith(".png"):

            im = Image.open('USPS_data/Numerals/'+str(i)+'/'+image_file_name)
            new_width = 28
            new_height = 28
            im = im.convert('1');
            im = im.resize((new_width, new_height), Image.ANTIALIAS)

            im.save('USPS_norm_data/Numerals/'+str(i)+'/' + image_file_name.replace('.png', '') + '_norm'+'.png')
