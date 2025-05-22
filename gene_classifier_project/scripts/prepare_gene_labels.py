import pandas as pd

# Load COSMIC
cosmic = pd.read_csv("data/Census_allThu May 22 21_50_23 2025.csv")
cosmic = cosmic[["Gene Symbol", "Role in Cancer"]].dropna()
cosmic["label"] = cosmic["Role in Cancer"].apply(
    lambda x: "TSG" if "TSG" in x else ("OG" if "oncogene" in x.lower() else None)
)
cosmic = cosmic.dropna(subset=["label"])
cosmic = cosmic.rename(columns={"Gene Symbol": "gene"})
cosmic = cosmic[["gene", "label"]]

# Load OncoKB
oncokb = pd.read_csv("data/cancerGeneList.tsv", sep="\t")
oncokb = oncokb[["Hugo Symbol", "Oncogene", "TumorSuppressorGene"]]
oncokb = oncokb.rename(columns={"Hugo Symbol": "gene"})
oncokb = oncokb.melt(id_vars=["gene"], value_vars=["Oncogene", "TumorSuppressorGene"])
oncokb = oncokb[oncokb["value"] == True]
oncokb["label"] = oncokb["variable"].apply(lambda x: "OG" if x == "Oncogene" else "TSG")
oncokb = oncokb[["gene", "label"]]

# Combine and deduplicate
combined = pd.concat([cosmic, oncokb]).drop_duplicates(subset=["gene"])

# Save to CSV
combined.to_csv("data/gene_labels.csv", index=False)
print(f"Saved {len(combined)} labeled genes to data/gene_labels.csv")
