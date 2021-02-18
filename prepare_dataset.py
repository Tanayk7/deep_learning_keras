from os import makedirs
from os import listdir
from shutil import copyfile, move
from random import seed
from random import random

dataset_home = 'Datasets/DogsVSCats/'
train_images_directory = dataset_home + "train_images"
subdirs = ['train/', 'test/']
src_directory = dataset_home + 'train/'

val_ratio = 0.25

print("creating subdirectories....")
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['dogs/', 'cats/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)
print("Done creating subdirectories")

# seed random number generator
seed(1)

print("Moving images....")
for file in listdir(train_images_directory):
    src = train_images_directory + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/' + file
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/' + file
    move(src, dst)

print("Done moving images")
