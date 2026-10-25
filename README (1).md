# medhub-exposure

Code and results for: Majjaj I, Anisser N. "Systemic Exposure and Governance-Mediated Resilience of a
Global Freight Hub: A Geodata-Driven Approach Applied to MedHub–Tanger Med" (Data Science for Transportation, submitted).

## Reproduce
1. Download the Geofabrik Morocco extract and filter the study area (osmium-tool):

       wget https://download.geofabrik.de/africa/morocco-latest.osm.pbf
       osmium extract -b -7.80,33.40,-4.80,35.95 morocco-latest.osm.pbf -o area.osm.pbf
       osmium tags-filter area.osm.pbf w/highway=motorway,motorway_link,trunk,trunk_link,primary,primary_link,secondary,secondary_link w/railway=rail -o medhub_network.osm.pbf

2. Convert to OSM XML and run (Python ≥3.10, `pip install osmnx networkx geopandas osmium matplotlib`):

       python -I pbf2xml.py medhub_network.osm.pbf        # writes roads.osm and rail.osm
       python hub_exposure.py --config config_medhub.json --out results
       python make_figures.py

Results in this repository were produced with the extract dated 2026-10-24.
The pre-computed result tables (`table_*.csv`, `network_summary.json`, `critical_links.gpkg`) and the figures (`fig3`–`fig6`) used in the paper are stored at the repository root; re-running the scripts writes them to `results/` and `figures/`.
Self-test without data: `python hub_exposure.py --synthetic --out test_results`.

OpenStreetMap data © OpenStreetMap contributors, ODbL.
