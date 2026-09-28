# Clinical-Image Experiments

The public fusion interface is `brits_mi.image_fusion.BRITSMIImage`. For a generated
data example, use the image command in the [simulation guide](../README.md).

| Component | Location | Output |
| --- | --- | --- |
| Joint clinical-image generator | `longitudinal_sim.data.generate_dataset` | Biomarker trajectories, masks, times, clinical covariates, grayscale images, and class labels |
| Pixel summaries | `brits_mi._image_reference.image_features` | Seventeen image features per subject |
| Image-conditioned imputation | `BRITSMIImage.complete` | One imputed trajectory per subject |
| Clinical feature extraction | `trajectory_motif_features` | Thirty-four longitudinal clinical summaries |
| Feature-level fusion | `BRITSMIImage.predict_proba` | Three-class probabilities from a fitted ExtraTrees classifier |
| Single-method generated-data run | `brits_mi.image_fusion.run_image_replicate` | Classification and imputation metrics |
| Historical comparison | `scripts/run_fast_image_fusion_nsim500.py` | Archived multi-method comparison implementation |

The public image method uses tree-based imputation, not recurrent stochastic
imputation. The historical forest comparator is a surrogate rather than R
missForest, and the historical arm named MICE has an input-encoding defect.
These distinctions are documented in [image implementation details](../../docs/image_fusion.md).
The historical script is retained for inspection; it should not be used to
report those arms as validated package-default comparisons.
