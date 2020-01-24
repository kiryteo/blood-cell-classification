from scipy.io import loadmat
from collections import defaultdict
import numpy as np
import os
import pickle as pk


def remove_similar(seq, k, similarity_function=lambda x, y: np.linalg.norm(x - y)):
    '''
    Take a list of numpy array and remove the images which are similar until the sequence has size k
    :param seq:
    :param k:
    :param similarity_function: Function used to evaluate similarity between 2 numpy arrays. Has signature
    np.array -> np.array -> Real
    :return:
    '''

    while len(seq) > k:
        min_dissim = np.inf
        argmin = -1

        for i in range(len(seq) - 1):
            dissim = similarity_function(seq[i], seq[i + 1])

            if dissim < min_dissim:
                min_dissim = dissim
            argmin = i

        seq = seq[:argmin] + seq[argmin + 1:]

    return seq


def extract_reduced(path, k):
    '''
    Apply remove_similar to each seq from the file specified un path

    :param path:
    :param k:
    :return: The sequence, into a list of 3 np.array: l[label] = np.array(N_sample, k)
    '''
    mat = loadmat(path)

    seqs = mat['Norm_Tab']
    labels = mat['Labels_Num']

    reduced_seqs = [[], [], []]

    for i in range(seqs.shape[0]):
        #print(f'{i}/{seqs.shape[0]}')
        reduced_seqs[labels[i][0]].append(remove_similar(list(seqs[i]), k))

    for i in range(3):
        reduced_seqs[i] = np.array(reduced_seqs[i])

    return reduced_seqs


def sized_sorted_seqs(path, trunc=None):
    '''
    Extract a mat file and format it with the following format: list label -> dict seq_size -> array of sequences

    all_sequences[label][seq_size] = array[N_samples, seq_size, 31, 31, 1]

    :param trunc: threshold to trunc the sequences
    :param path: path to the file
    :return: all_sequences
    '''
    mat = loadmat(path)

    seqs = mat['Norm_Tab']
    labels = mat['Labels_Num']

    # (3321, 67, 31, 31)
    # (3321, 1)

    all_sequences = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(seqs.shape[0]):
        s = []
        for j in range(trunc):
            if j < seqs[i].shape[0]:
                s.append(seqs[i, j])

        if s:
            all_sequences[labels[i][0]][len(s)] += [s]

    for i in range(3):
        for k in all_sequences[i].keys():
            all_sequences[i][k] = np.array(all_sequences[i][k])

    return all_sequences


def raw_sequences(path):
    mat = loadmat(path)

    seqs = mat['Norm_Tab']
    labels = mat['Labels_Num']

    X = [[], [], []]

    for i in range(seqs.shape[0]):
        s = []
        if len(X[labels[i][0]]) > 100:
            continue
        for j in range(seqs[i].shape[0]):
            if np.all(seqs[i, j] == np.zeros((31,31))):
                break
            s.append(seqs[i, j])

        if s is not None:
            X[labels[i][0]].append(s)

    del(mat)

    return X


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extracts the globule sequences from the mat files.')
    parser.add_argument('path', type=str, help='Path to the directory or file to process.')
    parser.add_argument('output', type=str, help='Output path, directory where all the files will be put if the input'
                                                 'is a directory or file if the input was a file.')

    parser.add_argument('-m', '--merge', default=False, action="store_true",
                        help='If a directory is specified as input, it will merge all the data in one single file.'
                             'Warning: It could use a lot of RAM.')

    parser.add_argument('-a', '--all', default=False, action="store_true",
                        help='Retrieve all the sequences'
                             'Warning: It could use a lot of RAM.')

    parser.add_argument('-t', '--trunc', type=int, default=None,
                        help='Take all the sequences, trunc them if their size exceed the specified value.')

    parser.add_argument('-s', '--similarity', type=int, default=None,
                        help='Remove images which are too similar until the sequence has the specified size.')

    args = parser.parse_args()

    if os.path.isdir(args.path):
        files = [os.path.join(args.path, f) for f in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, f))]
    else:
        files = [args.path] if os.path.isfile(args.path) else []

    if args.trunc and args.trunc > 0:
        if not args.merge and len(files) > 0:
            for i, f in enumerate(files):
                print(f)
                d = sized_sorted_seqs(f, trunc=args.trunc)
                pk.dump(d, open(args.output.format(i=i), "wb"), protocol=pk.HIGHEST_PROTOCOL)
        else:
            ret = None
            for f, i in enumerate(files):
                if i == 0:
                    ret = sized_sorted_seqs(f, trunc=args.trunc)
                else:
                    d = sized_sorted_seqs(f, trunc=args.trunc)
                    for i in range(3):
                        for k, v in ret[i].items():
                            if k in d[i]:
                                ret[i][k] = np.concatenate([v, d[i][k]])
                        for k, v in d[i].items():
                            if k not in ret[i]:
                                ret[i][k] = v

                    pk.dump(ret, open(args.output, "wb"), protocol=pk.HIGHEST_PROTOCOL)

    elif args.similarity and args.similarity > 0:
        if not args.merge and len(files) > 0:
            for i, f in enumerate(files):
                d = extract_reduced(f, k=args.similarity)
                pk.dump(d, open(args.output.format(i=i), "wb"))
        else:
            ret = None
            for i, f in enumerate(files):
                if i == 0:
                    ret = extract_reduced(f, k=args.similarity)
                else:
                    d = extract_reduced(f, k=args.similarity)
                    for i in range(3):
                        ret[i] = np.concatenate([ret[i], d[i]], axis=0)

            pk.dump(ret, open(args.output, "wb"), protocol=pk.HIGHEST_PROTOCOL)

    elif args.all:
        if not args.merge and len(files) > 0:
            for i, f in enumerate(files):
                d = raw_sequences(f)
                pk.dump(d, open(args.output.format(i=i), "wb"), protocol=pk.HIGHEST_PROTOCOL)
        else:
            ret = None
            for i, f in enumerate(files):
                print(i, len(files))
                if i == 0:
                    ret = raw_sequences(f)
                else:
                    d = raw_sequences(f)
                    for i in range(3):
                        ret[i] = ret[i] + d[i]

            pk.dump(ret, open(args.output, "wb"), protocol=pk.HIGHEST_PROTOCOL)






