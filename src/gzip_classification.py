#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gzip_classification.py - use gzip to classify protein sequences
author: Bill Thompson
license: GPL 3
copyright: 2023-07-17

See: “Low-Resource” Text Classification: A Parameter-Free Classification
      Method with Compressors by Jiang et al.
    https://aclanthology.org/2023.findings-acl.426/
    
"""
import gzip
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def read_fasta(filename: str) -> list[tuple[SeqRecord, str, int]]:
    """read a collection of sequences from a FASTA file.

    Parameters
    ----------
    filename : str
       The file containing human and yeast sequences in FASTA format

    Returns
    -------
    list[tupe[SeqRecord, str, int]]
        a list of tuples, (BioPython SeqRecord, sequence species, compressed length of sequence)
    """
    seq_recs = []
    for seq in SeqIO.parse(filename, "fasta"):
        if seq.id.find("HUMAN") != -1:
            seq_type = "HUMAN"
        else:
            seq_type = "YEAST"

        comp_len = len(gzip.compress(str(seq.seq).encode()))
        if comp_len < len(str(seq.seq)):
            seq_recs.append((seq, seq_type, comp_len))

    return seq_recs

def get_training_test_index(length: int, training_pct: float = 0.8) -> tuple[list, list]:
    """Create indices of sequences for training and testing

    Parameters
    ----------
    length : int
        length of sequence list 
    training_pct : float, optional
        proportion of data used for training, by default 0.8

    Returns
    -------
    tuple[list, list]
        list of indices for training and testing 
    """
    idx = np.random.choice(length, size = int(np.rint(training_pct * length)), replace = False)
    mask = np.full(length, True, dtype = bool)
    mask[idx] = False

    ids = np.array(list(range(length)), dtype = int)

    return list(ids[~mask]), list(ids[mask])

def predict(seq_recs: list[tuple[SeqRecord, str, int]], 
            training_idx: list[int], 
            test_idx: list[int],
            k: int = 2) -> list[str]:
    """
    Predict protein species using gzip compression

    Parameters
    ----------
    seq_recs : list[tuple[SeqRecord, str, int]]
        a list of tuples (BioPython SeqRecord, species name, compressed sequence length)
    training_idx : list[int]
        a list of integers indicating which sequences to use for training
    test_idx : list[int]
        a list of integers indicating which sequences to use for testing
    k : int, optional
        number of top comparisons to consider, by default 2

    Returns
    -------
    list[str]
        a list of species predictions

    Note
    ----
    The paper uses k = 2. In the code below, we use k = 1.
    """
    predictions = []

    # loop through the test set and compare each to the training sequences
    for ix in test_idx:
        (x , _, lx) = seq_recs[ix]
        dist = []
        for iy in training_idx:
            (y , _, ly) = seq_recs[iy]
            xy = " ".join([str(x.seq), str(y.seq)])
            lxy = len(gzip.compress(xy.encode()))
            dxy = (lxy - min(lx, ly)) / max(lx, ly)
            dist.append(dxy)
        
        idx = np.argsort(np.array(dist))
        top_k = [seq_recs[training_idx[i]][1] for i in idx[:k]]
        prediction = max(set(top_k), key = top_k.count)
        predictions.append(prediction)

    return predictions

def main():
    data_file = '/mnt/d/Documents/analytic_garden/hyperdim/data/sapiens_yeast_proteins.fasta'

    seq_recs = read_fasta(data_file)
    training_idx, test_idx = get_training_test_index(len(seq_recs))

    # make predictions and count them
    predictions = predict(seq_recs, training_idx, test_idx, k = 1)
    m = np.sum(np.array([predictions[i] == seq_recs[test_idx[i]][1] for i in range(len(predictions))]))
    print(m / len(predictions))

    # let's see how well we did with each species
    m2 = np.sum([(np.array(seq_recs[test_idx[i]][1] == "HUMAN") and 
                (predictions[i] == seq_recs[test_idx[i]][1])) for i in range(len(predictions))])
    humans_in_test_set = np.sum(np.array([seq_recs[test_idx[i]][1] == "HUMAN" for i in range(len(test_idx))]))
    print(m2, "correct human predictions out of", humans_in_test_set, m2 / humans_in_test_set)
    m3 = np.sum([(np.array(seq_recs[test_idx[i]][1] == "YEAST") and 
                (predictions[i] == seq_recs[test_idx[i]][1])) for i in range(len(predictions))])
    yeast_in_test_set = np.sum(np.array([seq_recs[test_idx[i]][1] == "YEAST" for i in range(len(test_idx))]))
    print(m3, "correct yeast predictions out of", yeast_in_test_set, m3 / yeast_in_test_set)
    print()

if __name__ == "__main__":
    main()
