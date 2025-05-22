import pandas as pd
import requests
import os
import time

# Load gene list
df = pd.read_csv("data/gene_labels.csv")
genes = df["gene"].unique()

# Output folder
os.makedirs("data/raw_sequences", exist_ok=True)

def get_ensembl_id(symbol):
    url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{symbol}?expand=1"
    headers = {"Content-Type": "application/json"}
    r = requests.get(url, headers=headers)
    if r.ok:
        return r.json()["id"]
    return None

def fetch_sequence(ensembl_id):
    url = f"https://rest.ensembl.org/sequence/id/{ensembl_id}?type=genomic"
    headers = {"Content-Type": "text/plain"}
    r = requests.get(url, headers=headers)
    return r.text if r.ok else None

# Fetch sequences
for gene in genes:
    print(f"Fetching {gene}...")
    ensembl_id = get_ensembl_id(gene)
    if ensembl_id:
        sequence = fetch_sequence(ensembl_id)
        if sequence:
            with open(f"data/raw_sequences/{gene}.txt", "w") as f:
                f.write(sequence)
            print(f"✅ Saved {gene}")
        else:
            print(f"❌ Failed to get sequence for {gene}")
    else:
        print(f"❌ Could not find Ensembl ID for {gene}")
    time.sleep(0.5)  # To avoid rate limits

print("✅ Done fetching all sequences.")
