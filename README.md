# hypinvalimap

Experiments and workflows for mapping invasive alien species using hyperspectral (HS) and multispectral (MS) data, change detection (IMAD/IRMAD), and Google Earth Engine (GEE).

## Repository structure
- code/: Jupyter notebooks and supporting assets.
  - catboost_info/: Auto-created logs/metadata from CatBoost training.
  - Figures/: Generated figures/animations used in reports.
  - LUCAS/: Ancillary data/scripts related to LUCAS processing.
- data/: Input/output datasets (shapefiles, GeoPackage, GeoJSON, Feather tables).

## Notebooks (what each does and how to run)
Open in Jupyter Lab/Notebook and run cells top-to-bottom. Adjust any paths to the data/ folder if needed.
- Basic_ML_pipeline.ipynb — Baseline ML workflow (feature prep, train/validate ML models such as CatBoost/Scikit-Learn) on HS/MS features.
- data_exploration.ipynb — Exploratory data analysis of spectra/labels; summary stats and plots.
- efm_NB.ipynb — Naive Bayes–style baseline experiments for classification.
- extract hs data.ipynb — Extracts hyperspectral samples/features to tabular data from vectors in data/ (produces Feather/GeoJSON outputs).
- GEE_imad_tutorial.ipynb — IMAD change detection in Google Earth Engine (part 1). Requires GEE auth.
- GEE_imad_tutorial_p2.ipynb — Continuation of IMAD tutorial/workflow in GEE (part 2). Requires GEE auth.
- HSI_experiments.ipynb — Hyperspectral modeling experiments (model selection, feature importance, metrics).
- HS_experiments_12082025.ipynb — Dated HS experiment run with specific configs/results.
- IAS_mapping_Example_GEEConformal.ipynb — IAS mapping in GEE with conformal prediction. Requires GEE auth.
- IRMAD_experiments.ipynb — IRMAD change detection experiments (e.g., 2018 vs 2023 comparisons).
- IRMAD_experiments_HS_case study.ipynb — IRMAD workflow focused on a hyperspectral case study.
- Labelling_tools.ipynb — Tools for creating/cleaning labels; edits/exports shapefiles/GeoJSON in data/.
- MS_experiments.ipynb — Multispectral modeling experiments.
- MS_experiments_01082025.ipynb — Dated MS experiment run with specific configs/results.
- plot_2018_2023_signatures.ipynb — Plots spectral/signature changes between 2018 and 2023 using extracted tables.
- sentinel2.gif — Example animation/illustration used in figures or presentations.

## Data folder contents (read-only inputs and derived outputs)
- 2018_2023_MgnChg.{shp,shx,dbf,prj,cpg} — Magnitude-of-change layer for 2018–2023.
- aliens_sep2018.{shp,shx,dbf,prj,cpg} — Alien species training/validation vectors (2018 snapshot).
- aliens_sep2018_bioscape2023.{shp,shx,dbf,prj,cpg} — 2018 labels intersected/updated with 2023 BioScape.
- aliens_sep2018.gpkg — GeoPackage version of alien species vectors.
- aliens_sep2018.qmd — Quarto notes/report associated with labeling/data prep.
- 2018_extracted_data.feather — HS/MS features extracted for 2018 samples.
- 2023_extracted_data.feather — HS/MS features extracted for 2023 samples.
- 2023_extracted.geojson — Feature subset for visualization/QA.

## How to run
- Requirements (typical): Python 3.10+, JupyterLab/Notebook, numpy, pandas, geopandas, scikit-learn, catboost, matplotlib, seaborn. For spatial I/O: rasterio, shapely, pyproj, fiona; optional: GDAL. For GEE notebooks: earthengine-api.
- Start: jupyter lab (or jupyter notebook), open a notebook in code/, set any base/data paths if needed, and run all cells.
- GEE setup (for GEE_* and IAS_mapping_* notebooks):
  1) pip install earthengine-api
  2) In a notebook: import ee; ee.Authenticate(); ee.Initialize()

## Notes
- Notebooks assume data/ is present at repo root; change paths to match your environment.
- Some notebooks write outputs into data/ or code/Figures/. Keep versions/dates in filenames for reproducibility.
- Coordinate reference systems must be consistent when mixing vector and raster data; reproject as needed.