# Literature Scraping Pipeline

Daily automated discovery of brain scRNA-seq and spatial transcriptomics papers/datasets.

## Sources

- **PubMed** — NCBI E-utilities (papers)
- **bioRxiv/medRxiv** — bioRxiv REST API (preprints)
- **GEO** — NCBI GEO DataSets (datasets)
- **Zenodo** — Zenodo REST API (datasets)
- **CellxGene** — CZI CellxGene Discover (datasets, optional)

## Setup

```bash
# Install dependencies
pip install -r literature/requirements.txt

# Set NCBI API key (get from https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)
export NCBI_API_KEY="your-key-here"  # add to ~/.zshrc

# Install daily cron (macOS)
cp literature/com.maxxyung.literature-scraper.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.maxxyung.literature-scraper.plist
```

## Usage

```bash
# Run scraper manually
python -m literature scrape
python -m literature scrape --source pubmed
python -m literature scrape --dry-run

# Review discovered papers
python -m literature review
python -m literature review --limit 20

# Check status
python -m literature status

# Export approved papers
python -m literature export --format csv --status approved
python -m literature export --format json --with-datasets
```

## Configuration

Edit `literature/config.yaml` to customize search queries, enable/disable sources, and adjust date ranges.
