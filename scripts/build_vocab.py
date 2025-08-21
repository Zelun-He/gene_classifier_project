from itertools import product
import os

def generate_vocab(k=6):
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    os.makedirs("data/vocab", exist_ok=True)
    with open("data/vocab/vocab.txt", "w") as f:
        for kmer in kmers:
            f.write(f"{kmer}\n")
    print(f"✅ Wrote {len(kmers)} k-mers to data/vocab/vocab.txt")

if __name__ == "__main__":
    generate_vocab(k=6)
