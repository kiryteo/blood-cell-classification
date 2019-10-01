import glob
import shutil

folders = glob.glob('/media/ashwin/Windows/blood-cells/files/*')

fd = folders[1:]

files = glob.glob('/media/ashwin/Windows/blood-cells/files/0csvs/*')

#print(files)

for each in fd:
    name = each.split('/')[-1]
    name = name[:2] + name[7:10] + name[-3:] + '.csv'
    name = '/media/ashwin/Windows/blood-cells/files/0csvs/' + name
    each = each + '/'
    # print(each)
    # print(name)
#    break
    if name in files:
        print(name)
        print(each)
        shutil.copy(name, each)