# Mapping Bee-Friendly Trees Using Remote Sensing

This repository contains the code and data used for the study **"Mapping of Bee-Friendly Trees Through Remote Sensing: A Novel Approach to Enhance Pollinator Conservation"**, as published by Robbe Neyns *et al.* (2025). The project explores the use of deep learning and remote sensing data to map *Salix* (willow) trees, a key pollen source for the solitary bee *Andrena vaga*, within the city of Braunschweig, Germany.

## Overview

The core of this project is a deep learning pipeline that maps *Salix* trees by processing multi-temporal PlanetScope imagery with a tabular transformer model, SAINT (Self-Attention and Intersample Attention Transformer). This mapping is then used to analyze the relationship between the spatial distribution of *Salix* trees and the nesting behavior of *A. vaga*.

## Key Features

- 🌳 **Tree Segmentation**: Trees were segmented from high-resolution airborne LiDAR data using a local maxima and marker-controlled watershed approach.
- 🛰️ **Spectral Data Integration**: PlanetScope time series imagery (40 images) was extracted over segmented crowns to provide phenological profiles.
- 🤖 **Deep Learning Model**: SAINT was used for tree classification based on multi-temporal, multi-spectral tabular input data.
- 🐝 **Ecological Analysis**: Spatial relationships between *Salix* trees and bee nest aggregations were analyzed, including pollen provisioning estimates and foraging distances.

## Model Details

- **Architecture**: SAINT (Somepalli et al., 2021)
- **Input**: 40 × 8 spectral band reflectances per tree (uncalibrated)
- **Embedding**: Day of year and band number embeddings based on Grigsby et al. (2021)
- **Handling Imbalance**: Oversampling of the minority class (*Salix*)
- **Batch Size**: Optimal performance at 126
- **Validation**: Stratified sampling (10% of the dataset)

## Performance

| Metric            | Value |
|-------------------|--------|
| Accuracy          | 98.8%  |
| AUROC             | 0.97   |
| Precision (Salix) | 0.69   |
| Recall (Salix)    | 0.78   |
| F1-score          | 0.73   |


## Citation

If you use this code, please cite the paper:

> Neyns, R., Gardein, H., Muenzinger, M., Hecht, R., Greil, H., & Canters, F. (2025). Mapping of bee-friendly trees through remote sensing: a novel approach to enhance pollinator conservation. *Ecological Informatics*.

## Acknowledgements

This work was supported by the European Space Agency (ESA) and Planet Labs through the ESA Third Party Mission Project (PP0089119). LiDAR and aerial data were provided by the city of Braunschweig.
