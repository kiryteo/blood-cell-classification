import os
import shutil
import sys
import glob


def dir_creator():
    dirr = glob.glob('/media/ashwin/Windows/blood-cells/files/*')

    for each in dirr:
        os.chdir(each)
        if not os.path.exists('0'):
            os.makedirs('0')
        if not os.path.exists('1'):
            os.makedirs('1')

def move_files():
    directories = glob.glob('/media/ashwin/Windows/blood-cells/files/*')
    directories = directories[:-1]

    for each_dir in directories:
        os.chdir(each_dir)
        csvname = each_dir.split('/')[-1]
        csvname = csvname[:2] + csvname[7:10] + csvname[-3:] + '.csv'
        if not os.listdir('0'):
            with open(csvname) as cfile:
                for lines in cfile:
                    lines = lines.strip()
                    file = lines.split(',')[1]
                    label = lines.split(',')[2]
                    #print(label)
                    if label == '0':
                        path = os.getcwd()
        #                   print(path)
                        shutil.move(file, path + '/0/')

        if not os.listdir('1'):
            with open(csvname) as cfile:
                for lines in cfile:
                    lines = lines.strip()
                    file = lines.split(',')[1]
                    label = lines.split(',')[2]
                    if label == '1':
                        path = os.getcwd()
        #                   print(path)
                        shutil.move(file, path + '/1/')

        if not os.path.exists('2'):
            os.makedirs('2')
            with open(csvname) as cfile:
                for lines in cfile:
                    lines = lines.strip()
                    file = lines.split(',')[1]
                    label = lines.split(',')[2]
                    if label == '2':
                        path = os.getcwd()
                        shutil.move(file, path + '/2/')
        else:
            if not os.listdir('2'):
                with open(csvname) as cfile:
                    for lines in cfile:
                        lines = lines.strip()
                        file = lines.split(',')[1]
                        label = lines.split(',')[2]
                        if label == '2':
                            path = os.getcwd()
                            shutil.move(file, path + '/2/')


def move_to_final():

    files = glob.glob('/media/ashwin/Windows/blood-cells/files/*')
    for each in files:
        print(each)
        os.chdir(each)
        dirr0 = each + '/0/*'
        newdir0 = glob.glob(dirr0)
        for file in newdir0:
            shutil.copy(file, '/media/ashwin/Windows/blood-cells/finaldata/0/')
        dirr1 = each + '/1/*'
        newdir1 = glob.glob(dirr1)
        for file in newdir1:
            shutil.copy(file, '/media/ashwin/Windows/blood-cells/finaldata/1/')
        dirr2 = each + '/2/*'
        newdir2 = glob.glob(dirr2)
        for file in newdir2:
            shutil.copy(file, '/media/ashwin/Windows/blood-cells/finaldata/2/')

def main():
    #dir_creator()
    #move_files()
    move_to_final()

if __name__ == '__main__':
    main()