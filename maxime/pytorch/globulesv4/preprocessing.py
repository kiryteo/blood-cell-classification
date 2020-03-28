from scipy.io import loadmat
from collections import defaultdict
import numpy as np
import os
import pickle as pk
import argparse

def raw_sequences(path):
    mat = loadmat(path)

    seqs = mat['Norm_Tab']
    labels = mat['Labels_Num']

    # labels[i][0]
    # seqs[i, j] = image j of seq i

    X = []
    Y = []

    for i in range(seqs.shape[0]):
        s = []
        for j in range(seqs[i].shape[0]):
            if j > 64:
                break
            if np.all(seqs[i, j] == np.zeros((31,31))):
                break
            s.append(seqs[i, j])

        if s is not None:
            X.append(np.array(s))
            Y.append(np.array(labels[i][0]))

    del(mat)

    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts the globule sequences from the mat files.')
    parser.add_argument('path', type=str, help='Path to the directory or file to process.')
    parser.add_argument('output', type=str, help='Output path, directory where all the files will be put if the input'
                                                 'is a directory or file if the input was a file.')

    parser.add_argument('-m', '--merge', default=False, action="store_true",
                        help='If a directory is specified as input, it will merge all the data in one single file.'
                             'Warning: It could use a lot of RAM.')

    args = parser.parse_args()

    if os.path.isdir(args.path):
        files = [os.path.join(args.path, f) for f in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, f))]
    else:
        files = [args.path] if os.path.isfile(args.path) else []

    if not args.merge and len(files) > 0:
        for i, f in enumerate(files):
            print(f)
            d = raw_sequences(f)
            pk.dump(d, open(args.output.format(i=i), "wb"), protocol=pk.HIGHEST_PROTOCOL)

    elif args.merge:
        X = []
        Y = []

        for i, f in enumerate(files):
            print(f)
            X_tmp,Y_tmp = raw_sequences(f)
            X += X_tmp
            Y += Y_tmp

        pk.dump((X,np.array(Y)), open(args.output, "wb"), protocol=pk.HIGHEST_PROTOCOL)


