# fMRI Simulation of Visual Pathways

---

## Project Overview

This project aims to simulate the results observed in our fMRI experiment, where distinct patterns of feature representation emerge along the ventral and dorsal visual pathways. Our goal is to model these findings using convolutional neural networks (CNNs), incorporating both a baseline model and an advanced model with recurrency to replicate the observed phenomena.

## Research Background

In our original fMRI experiment, we identified the following key findings:

### Ventral Visual Pathway
- **Feature-Specific Representations**: Along this pathway, average feature representations such as orientation, spikiness, and animacy emerge.
- **Hierarchical Segregation**: These representations are hierarchically segregated, meaning different levels of the visual hierarchy encode distinct features in progressively abstract ways.

### Dorsal Visual Pathway
- **Feature-Agnostic Representations**: Abstract ratio representations, such as the proportion of items in an ensemble, emerge in this pathway.
- **Cross-Decoding Evidence**: The feature-agnostic nature of this pathway is evidenced by above-chance cross-decoding performance between different features (e.g., decoding animacy using models trained on orientation).

## Simulation Goals

This project seeks to replicate the above findings computationally by designing and training neural network models to:
1. Simulate **feature-specific hierarchical segregation** in the ventral pathway.
2. Model **feature-agnostic abstract ratio representations** in the dorsal pathway, validated through cross-decoding analyses.

## Planned Approach

1. **Baseline Model**:
   - A standard convolutional neural network (CNN) to simulate hierarchical feature representations in the ventral visual pathway.

2. **Advanced Model with Recurrency**:
   - Incorporates recurrent connections to model temporal dynamics that may contribute to feature-agnostic ratio representation in the dorsal pathway.

3. **Data Simulation**:
   - Synthetic datasets with controlled features such as orientation, spikiness, animacy, and ratios will be generated to emulate stimuli from the fMRI experiment.

4. **Analysis**:
   - Feature-specific decoding accuracy to evaluate ventral pathway representations.
   - Cross-decoding performance to assess dorsal pathway representations.

## Key Questions

1. **Can CNNs replicate the hierarchical feature-specific representations observed in the ventral visual pathway?**
2. **Can recurrent mechanisms enable feature-agnostic ratio representations akin to those found in the dorsal visual pathway? How robust are these representations to cross-decoding across different feature dimensions?**
3. **Are we gonna see a ventral and dorsal like bifurcation naturally emerge in the networks?**

## Future Directions

- **Explore biologically inspired mechanisms**: such as a retinotopic presentation of the image in smaller patches and sequentially 

## Contributors

- **Patxi Elosegi**  
- **Alejandro Tabas**  
