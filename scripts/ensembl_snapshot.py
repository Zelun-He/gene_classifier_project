"""Pinned Ensembl 114 transport with hashed, replayable response caching."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REST = "https://may2025.rest.ensembl.org"
BIOMART = "https://may2025.archive.ensembl.org/biomart/martservice"
RELEASE = 114
ASSEMBLY = "GRCh38.p14"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


class SnapshotClient:
    def __init__(self, cache_dir, offline=False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.used = {}
        self.last_request = 0.0
        self.rate_lock = threading.Lock()

    def text(self, url, body=None, refresh=False):
        request_body = None if body is None else json.dumps(body, sort_keys=True, separators=(",", ":"))
        specification = {"url": url, "body": request_body}
        key = digest(json.dumps(specification, sort_keys=True).encode())
        path = self.cache_dir / f"{key}.json"
        if path.exists() and (self.offline or not refresh):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["request"] != specification or digest(record["response"].encode()) != record["sha256"]:
                raise ValueError(f"Corrupt cached Ensembl response: {path}")
        else:
            if self.offline:
                raise ValueError(f"Offline cache miss for {url}")
            for attempt in range(4):
                with self.rate_lock:
                    time.sleep(max(0, 0.22 - (time.monotonic() - self.last_request)))
                    self.last_request = time.monotonic()
                try:
                    request = Request(url, data=None if request_body is None else request_body.encode(), headers={
                        "Content-Type": "application/json", "Accept": "application/json",
                        "User-Agent": "gene-classifier-sequence-audit/1.0",
                    })
                    with urlopen(request, timeout=40) as response:
                        payload = response.read()
                        record = {"request": specification, "response": payload.decode("utf-8"),
                                  "sha256": digest(payload), "final_url": response.url}
                    break
                except HTTPError as error:
                    if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                        raise RuntimeError(f"Ensembl HTTP {error.code}: {url}") from error
                    retry = error.headers.get("Retry-After", "")
                    time.sleep(min(15, float(retry) if retry.replace('.', '', 1).isdigit() else 2 ** attempt))
                except (URLError, TimeoutError) as error:
                    if attempt == 3:
                        raise RuntimeError(f"Ensembl request failed: {url}: {error}") from error
                    time.sleep(2 ** attempt)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
        self.used[key] = {"url": url, "request_body": request_body, "response_sha256": record["sha256"]}
        return record["response"]

    def rest(self, endpoint, body=None, refresh=False):
        return json.loads(self.text(REST + endpoint, body, refresh))

    def reference(self, refresh=False):
        release = self.rest("/info/data", refresh=refresh)
        assembly = self.rest("/info/assembly/homo_sapiens", refresh=refresh)
        if release.get("releases") != [RELEASE] or assembly.get("assembly_name") != ASSEMBLY:
            raise ValueError(f"Reference mismatch: expected Ensembl {RELEASE}, {ASSEMBLY}")
        assembly["top_level_region"] = sorted(assembly["top_level_region"], key=lambda row: row["name"])
        return release, assembly

    def map_entrez(self, identifiers):
        mapping = {identifier: set() for identifier in identifiers}
        for offset in range(0, len(identifiers), 200):
            batch = identifiers[offset:offset + 200]
            if any(not identifier.isdigit() for identifier in batch):
                raise ValueError("BioMart mapping requires numeric Entrez IDs")
            query = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" datasetConfigVersion="0.6">'
                '<Dataset name="hsapiens_gene_ensembl" interface="default">'
                f'<Filter name="entrezgene_id" value="{",".join(batch)}"/>'
                '<Attribute name="ensembl_gene_id"/><Attribute name="entrezgene_id"/>'
                '</Dataset></Query>'
            )
            response = self.text(BIOMART + "?" + urlencode({"query": query}))
            for line in response.splitlines():
                if not line.strip():
                    continue
                columns = line.split("\t")
                if len(columns) != 2 or not columns[0].startswith("ENSG") or columns[1] not in batch:
                    raise ValueError(f"Unexpected BioMart response: {line[:160]}")
                mapping[columns[1]].add(columns[0])
        return mapping

    def lookup_genes(self, identifiers):
        result = {}
        batches = [identifiers[offset:offset + 20] for offset in range(0, len(identifiers), 20)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = pool.map(lambda batch: self.rest("/lookup/id?expand=1;mane=1", {"ids": batch}), batches)
            for batch, response in zip(batches, responses):
                if not isinstance(response, dict):
                    raise ValueError("Expected lookup dictionary from Ensembl")
                result.update({identifier: response.get(identifier) for identifier in batch})
        return result

    def sequences(self, queries, kind):
        result = {}
        endpoint = "/sequence/id?type=cds" if kind == "cds" else "/sequence/region/homo_sapiens?coord_system_version=GRCh38"
        batches = [queries[offset:offset + 40] for offset in range(0, len(queries), 40)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = pool.map(lambda batch: self.rest(endpoint, {"ids" if kind == "cds" else "regions": batch}), batches)
            for batch, response in zip(batches, responses):
                if not isinstance(response, list):
                    raise ValueError("Expected sequence list from Ensembl")
                for record in response:
                    query = record.get("query", record.get("id"))
                    if query not in batch or query in result:
                        raise ValueError(f"Unrequested or duplicate sequence response: {query}")
                    result[query] = record
        return result
