import sys
import glob
import os
from PIL import Image
import scipy.io

glist = glob.glob('/media/ashwin/Windows/blood-cells/files/*')

for each in glist:
    os.chdir(each)
    dirname = each.split('/')[-1]
    fname = each + '/' + dirname + '_Export.mat'

    mat = scipy.io.loadmat(fname)
    v1 = mat['Norm_Tab']

    for each_image in range(1, v1.shape[0]+1):
        imglist = []
        name = 'file' + str(each_image) + '-'
        for j in range(1,13):
            imglist.append(name + str(j) + '.png')

        for image in range(len(imglist)):
            delfile = each + '/' + imglist[image]
            if os.path.getsize(delfile) == 74:
                final_list = imglist[:image]
                break

        flist = []
        for final in final_list:
            final = each + '/' + final
            flist.append(final)

#   def combine_images():
        images = list(map(Image.open, flist))#['file11.png','file12.png','file13.png','file14.png','file15.png','file16.png'])
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]

        savename = dirname[7:10] + dirname[17:]
        savename = savename + str(each_image) + '.jpg'
        new_im.save(savename)


