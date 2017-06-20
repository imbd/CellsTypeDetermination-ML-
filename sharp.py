from PIL import Image, ImageFilter
import os

# folder = 'Data/Control/'
# new_folder = 'Data/Control(sharp)/'
folder = 'Old/Old(c)/'
new_folder = 'Old/Old(c)(sharp)/'

num = 0
for file in os.listdir(folder):
    image_file = os.path.join(folder, file)
    num += 1
    im = Image.open(image_file)
    im = im.filter(ImageFilter.SHARPEN)
    im = im.filter(ImageFilter.SHARPEN)
    im.save(new_folder + str(num) + '_01m.tif', 'TIFF')
    # im.save(new_folder + 'mod_' + file, 'TIFF')

'''
im_sharp = im.filter(ImageFilter.SHARPEN)
im_sharp = im_sharp.filter(ImageFilter.CONTOUR)
im_sharp = im_sharp.filter(ImageFilter.SHARPEN)
im_sharp = im_sharp.filter(ImageFilter.EDGE_ENHANCE)

'''
