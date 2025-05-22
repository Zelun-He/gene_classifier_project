def kmer_tokenize(seq, k=6):
    seq = seq.upper().replace("\n", "").replace(" ", "")
    return ' '.join([seq[i:i+k] for i in range(len(seq) - k + 1)])
