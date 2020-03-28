from matplotlib import pyplot as plt
import numpy as np


def write_seq(path, seq, true_label, preds):

    cols = ["True label", "0", "1", "2"]
    rows = []

    cell_text = [[str(true_label)] + [str(x) for x in preds]]

    n_seq = []
    for k, img in enumerate(seq):
        if np.all(img == np.zeros((31,31,1))):
            break
        n_seq.append(np.reshape(img, (31,31)))

    fig = plt.figure(figsize=(len(n_seq)+1 ,1), dpi=100)

    for k, img in enumerate(n_seq):
        fig.add_subplot(1, len(n_seq)+1, k+1)
        plt.imshow((img), cmap='gray', vmin=0, vmax=255)
        plt.axis('off')

    fig.add_subplot(1, len(n_seq) + 1, len(n_seq)+1)
    plt.axis('off')
    plt.table(cellText=cell_text, colLabels=cols)
    plt.savefig(path)
    plt.clf()










