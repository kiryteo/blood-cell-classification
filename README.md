# blood-cell-classification

Currently the data-format directory contains the script used to get the data from .mat files to the following structure for image classification.

```
 Dataset
    ||______
    |       |
 class_0 class_1 class_2
    |       |       |
    |       |       |_____________________________
    |       | ___________________                 |
    |                            |                |
samples for class_0   samples for class_1    samples for class_2

```

There are currently 6174 samples for class 0, 3079 samples for class 1 and 84,728 (more can be obtained) samples for class 2.

## Next steps:
1. Split the data into training, validation and test sets.
2. Approach-1 - Go with resizing of the images at run time and feed to CNN with fully connected layers
        1.1] Transfer learning with pretrained models
        1.2] Specific model for our case.
3. Approach-2 - Use Fully convolutional network which doesn't require similar image size as input to the network and proceed accordingly based on results.

## Issues with .mat files
Some of the files can't be processed using scipy, though these are very few and left out of the current data, we might need these later for the class 0 and class 1 samples.


TODO: Add the .mat files list
