# Literature Scrapers + Review Queue + Daily Cron — Design Spec

*Sub-project 1 of the Literature Intelligence Pipeline*
*Date: 2026-03-15*

## Overview

Daily automated pipeline that scrapes PubMed, bioRxiv/medRxiv, GEO, Zenodo, and CZI CellxGene for brain single-cell and spatial transcriptomics papers/datasets. Results go into a SQLite database. User reviews pending items via CLI. Runs as a macOS launchd cron job.

## Architecture

```
literature/
├── __init__.py
├── config.yaml                  # Search queries, sources, API keys ref, schedule
├── models.py                    # SQLAlchemy: Paper, Dataset, SearchRun tables
├── db.py                        # SQLite connection + session factory
├── scrapers/
│   ├── __init__.py
│   ├── base.py                  # BaseScraper ABC
│   ├── pubmed.py                # NCBI Entrez (papers)
│   ├── biorxiv.py               # bioRxiv + medRxiv REST API (papers)
│   └── dataset_sources.py       # GEO + Zenodo + CellxGene (datasets)
├── scheduler.py                 # Load config → run scrapers → dedupe → store
├── review.py                    # CLI: show pending, approve/reject/skip
├── cli.py                       # python -m literature {scrape,review,status,export}
├── requirements.txt
└── tests/
    ├── test_scrapers.py
    └── test_models.py
```

## Data Model (SQLite)

### Paper
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | Auto-increment |
| doi | TEXT UNIQUE | Dedup key |
| pmid | TEXT | PubMed ID (nullable) |
| title | TEXT NOT NULL | |
| authors | TEXT | Comma-separated |
| journal | TEXT | |
| year | INTEGER | |
| abstract | TEXT | |
| source | TEXT | pubmed, biorxiv, medrxiv |
| url | TEXT | |
| has_scrna_seq | BOOLEAN | Detected from abstract keywords |
| has_spatial | BOOLEAN | Detected from abstract keywords |
| species | TEXT | human, mouse, etc. |
| brain_regions | TEXT | JSON array |
| cell_count | INTEGER | If mentioned in abstract |
| accession_numbers | TEXT | JSON: {geo: [], zenodo: [], cellxgene: []} |
| status | TEXT | pending, approved, rejected, processed |
| discovered_at | DATETIME | |
| reviewed_at | DATETIME | Nullable |

### Dataset
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| paper_id | INTEGER FK | Nullable (datasets can be discovered independently) |
| accession | TEXT UNIQUE | GEO/Zenodo/CellxGene ID |
| source | TEXT | geo, zenodo, cellxgene |
| title | TEXT | |
| species | TEXT | |
| cell_count | INTEGER | |
| format | TEXT | h5ad, mtx, rds, loom |
| download_url | TEXT | |
| size_bytes | INTEGER | |
| status | TEXT | pending, approved, downloaded, qc_passed |
| discovered_at | DATETIME | |

### SearchRun
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| source | TEXT | |
| query | TEXT | |
| n_results | INTEGER | Total results from API |
| n_new | INTEGER | New papers/datasets added |
| timestamp | DATETIME | |

## Scraper Specifications

### BaseScraper (ABC)
```python
class BaseScraper(ABC):
    name: str
    rate_limit_delay: float  # seconds between requests

    @abstractmethod
    def search(self, query: str, max_results: int, date_from: date) -> list[PaperResult | DatasetResult]: ...

    def _rate_limit(self): time.sleep(self.rate_limit_delay)
```

### PubMed Scraper
- **API**: NCBI E-utilities (Entrez via biopython)
- **Auth**: API key via env var `NCBI_API_KEY` (user has one)
- **Rate limit**: 0.1s (10 req/sec with key)
- **Search**: `esearch` → `efetch` in XML, parse title/authors/abstract/DOI/PMID
- **Date filter**: `mindate`/`maxdate` params
- **scRNA-seq detection**: Check abstract for keywords (single-cell, scRNA-seq, snRNA-seq, 10x Genomics, Drop-seq)

### bioRxiv Scraper
- **API**: `https://api.biorxiv.org/details/biorxiv/{from}/{to}/{cursor}`
- **Auth**: None needed
- **Rate limit**: 1.0s
- **Search**: Fetch recent papers in neuroscience category, filter by keyword match in title/abstract
- **Also covers**: medRxiv (same API, different server param)

### Dataset Sources (merged: GEO + Zenodo + CellxGene)
- **GEO**: NCBI E-utilities `esearch` on GDS database, filter `Homo sapiens[Organism] AND brain[Title]`
- **Zenodo**: REST API `GET /api/records?q=...&type=dataset` (existing pattern from `00_zenodo_download.py`)
- **CellxGene**: `cellxgene-census` Python API, filter by `tissue_general == "brain"`
- **Rate limit**: 0.5s for GEO/Zenodo, 1.0s for CellxGene

## Config (YAML)

```yaml
db_path: literature/papers.db

sources:
  pubmed:
    enabled: true
    queries:
      - "brain organoid scRNA-seq"
      - "neural organoid single-cell"
      - "fetal brain atlas single cell"
      - "spatial transcriptomics brain development"
      - "cerebral organoid differentiation protocol"
    max_results_per_query: 100
    date_range_days: 7

  biorxiv:
    enabled: true
    queries:
      - "brain organoid"
      - "neural organoid scRNA"
      - "fetal brain single cell"
      - "cerebral organoid"
    server: biorxiv  # also supports: medrxiv
    date_range_days: 7

  datasets:
    geo:
      enabled: true
      queries:
        - "brain organoid Homo sapiens scRNA-seq"
        - "neural organoid single cell RNA"
    zenodo:
      enabled: true
      queries:
        - "brain organoid scRNA-seq h5ad"
        - "neural organoid atlas"
        - "fetal brain single cell"
    cellxgene:
      enabled: true
      tissue_filter: "brain"
```

## CLI Interface

```bash
# Daily scrape (cron calls this)
python -m literature scrape
python -m literature scrape --source pubmed  # single source
python -m literature scrape --dry-run        # show what would be fetched

# Review pending papers
python -m literature review                  # interactive triage
python -m literature review --limit 20       # review up to 20

# Status dashboard
python -m literature status                  # counts by status/source

# Export
python -m literature export --format csv --status approved
python -m literature export --format json --with-datasets
```

### Review CLI Flow
```
[1/15] "Single-cell atlas of human fetal cerebellum" (2026)
       Authors: Smith J, Chen L, ...
       Journal: Nature Neuroscience
       DOI: 10.1038/s41593-026-xxxxx
       Abstract: We present a comprehensive single-cell atlas of the
       developing human cerebellum spanning 8-22 gestational weeks...
       Datasets: GEO:GSE999999 (h5ad, 450K cells)

       [a]pprove  [r]eject  [s]kip  [q]uit  >
```

## Deduplication

1. **By DOI**: Primary dedup key. If DOI exists in DB, skip.
2. **By title similarity**: For preprints without DOI, use fuzzy title matching (Levenshtein ratio > 0.9).
3. **Preprint → publication**: When a paper with matching DOI appears from a journal source, update the existing preprint record (keep original `discovered_at`, update `journal`, `year`, `url`).

## Scheduling (launchd)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" ...>
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.maxxyung.literature-scraper</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/maxxyung/Projects/morphogen-gpbo/.venv/bin/python</string>
    <string>-m</string>
    <string>literature</string>
    <string>scrape</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/Users/maxxyung/Projects/morphogen-gpbo</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>6</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>/Users/maxxyung/Projects/morphogen-gpbo/literature/scrape.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/maxxyung/Projects/morphogen-gpbo/literature/scrape.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>NCBI_API_KEY</key>
    <string>${NCBI_API_KEY}</string>
  </dict>
</dict>
</plist>
```

## Dependencies

```
# literature/requirements.txt
requests>=2.31
pyyaml>=6.0
sqlalchemy>=2.0
biopython>=1.83
python-Levenshtein>=0.25  # fuzzy title matching
```

No heavy deps (no torch, scanpy, GPU). CellxGene Census is optional — gracefully skip if not installed.

## Testing

- `test_scrapers.py`: Mock API responses for each scraper, verify parsing
- `test_models.py`: SQLite CRUD operations, deduplication logic, status transitions

## What This Does NOT Do

- No PDF downloading or OCR (sub-project 2)
- No knowledge extraction from papers (sub-project 3, with scExtract)
- No vector embeddings (sub-project 4, with GenePT/CellWhisperer)
- No dataset downloading or QC (sub-project 5)
- No modification to the gopro/ pipeline

## Connection to Future Sub-projects

| Sub-project | Reads from | Adds to |
|---|---|---|
| 2: OCR/extraction | Papers with status=approved | Paper text, extracted protocols |
| 3: Knowledge graph | Extracted data from sub-project 2 | Vector embeddings, relationships |
| 4: Auto-discovery | Datasets with status=approved | Downloaded files, QC results |
