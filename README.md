# blood-cell-classification

Currently the data-format directory contains the script used to get the data from .mat files to the following structure for image classification.

```
  Dataset
    | | |____________________
    | |______________       |
    |               |       |
 Tank_treading   others    garbage
    |               |       |
    |               |       |_____________________________
    |               |_____________                        |
    |                            |                        |
samples for 'Tank_treading'   samples for 'others'    samples for 'garbage'

```

The samples with size less than 6 are removed (found only in 'garbage' class). For remaining samples, image trajectories are generated based on the minimum similarity algorithm along with SSIM function from sklearn.

## Experimental setup:
The image sample size generated from the old data: 13572 samples for 'Tank treading', 3924 samples for 'others', and 184814 samples for garbage (25000 randomly selected samples used for experiments).

Old data experimental setup:
```

                  train       valid          test
Tank treading      9000        2280          2292
others             2600         644           680
garbage           15000        5000          5000

```

The image sample size generated from the new data: 5186 samples for 'Tank treading', 838 samples for 'others', and 138236 samples for garbage (10000 randomly selected samples used for experiments).

New data experimental setup:
```

                  train       valid          test
Tank treading      3186        1000          1000
others              610         114           114
garbage            6000        2000          2000

```

## Directories and content:
1. TL-complete directory contains the notebooks for experiments with training on old data with fine-tuning in pretrained models (ResNet-152 architecture), saving the model and train for new data. The notebooks are present for k=[8,9] (k->trajectory size).
2. data-format directory contains the data processing codes used initially for old data. These include pipeline for the processing of .mat files to finally obtain the image trajectories.
3. fine-tuning directory contains the notebooks for experiments with training directly on new data with fine-tuning and using pretrained models (ResNet-152).
4. init-notebooks directory contains the notebooks for initial results based on np.resize method used for data processing and trajectory creation. All results for the old data. The method works to some degree but should be avoided since the images are either squeezed or enlarged based on the value of k.
5. resize-algo-notebooks directory contains the results for old data with new resizing algorithm mentioned above applied, just with the pretrained model and fine-tuning approach.

