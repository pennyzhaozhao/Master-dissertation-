# AI-Generated Facial Media Detection and Explainability

This repository contains the code used for a dissertation project on binary
classification of authentic and manipulated facial media. The project compares
traditional HOG-based machine-learning classifiers with deep-learning models,
then adds explainability analysis and a small local forensic demonstration tool.

The binary label mapping used throughout the project is:

- `0`: Real
- `1`: Manipulated

The Manipulated class is treated as the positive class for precision, recall,
F1-score and ROC-AUC reporting.

## Project Overview

Four classifiers are implemented and evaluated:

- **KNN + HOG**: K-Nearest Neighbours using Histogram of Oriented Gradients
  features.
- **Linear SVM + HOG**: linear SVM implemented with `SGDClassifier(loss="hinge")`
  using the same HOG feature pipeline.
- **CNN**: a compact convolutional neural network trained end-to-end on images.
- **ViT**: a fine-tuned Vision Transformer based on
  `google/vit-base-patch16-224`.

The main experimental aim is to compare classification performance and examine
which image regions influence each model's predictions.

## Repository Structure

```text
.
|-- prepare_dataset.py
|-- create_balanced_splits.py
|-- train_knn_hog.py
|-- train_linear_svm_hog.py
|-- train_svm_hog.py
|-- train_cnn_gradcam.py
|-- train_vit.py
|-- explainability_comparison.py
|-- README_explainability_comparison.md
`-- forensic_tool/
    |-- app.py
    |-- README.md
    `-- static/
        `-- style.css
```

## Main Scripts

| File | Purpose |
| --- | --- |
| `prepare_dataset.py` | Scans raw media, infers labels, extracts image/video frames, and creates train/validation/test CSV splits. |
| `create_balanced_splits.py` | Creates balanced train/validation/test CSV files by downsampling the majority class. |
| `train_knn_hog.py` | Trains and evaluates the KNN classifier using HOG features and a fitted `StandardScaler`. |
| `train_linear_svm_hog.py` | Trains and evaluates the linear SVM HOG baseline using mini-batch `SGDClassifier`. |
| `train_svm_hog.py` | Earlier SVM baseline script. |
| `train_cnn_gradcam.py` | Trains/evaluates the CNN and generates CNN Grad-CAM visualisations. |
| `train_vit.py` | Fine-tunes/evaluates the ViT classifier. |
| `explainability_comparison.py` | Generates cross-model occlusion sensitivity maps for KNN, linear SVM, CNN and ViT. |
| `forensic_tool/app.py` | Local web demo for uploading images/videos and detecting possible manipulation using the saved ViT model. |

## Model Artifacts

The code expects trained model artifacts in the following default locations:

```text
outputs/models/knn_hog_model.joblib
outputs/models/knn_hog_scaler.joblib
outputs/models/linear_svm_hog_model.joblib
outputs/models/linear_svm_hog_scaler.joblib
outputs/cnn/models/cnn_best_model.pt
outputs/vit/models/vit_best_model/
```

The ViT directory should contain:

```text
config.json
model.safetensors
preprocessor_config.json
training_state.pt
```

Large model checkpoints and datasets are not included in a lightweight GitHub
code submission unless Git LFS or an external download link is used.

## Dataset Splits

The training scripts use CSV split files generated during preprocessing. The
balanced split files are expected at:

```text
outputs/dataset/train_balanced.csv
outputs/dataset/val_balanced.csv
outputs/dataset/test_balanced.csv
```

The CNN and ViT scripts also save filtered CSV files after removing missing or
unreadable images:

```text
outputs/cnn/filtered_csvs/
outputs/vit/filtered_csvs/
```

All reported test metrics in the dissertation are based on the independent test
partition. The train/validation/test split should not be altered when reviewing
or reproducing the reported results.

## Running the Model Training Scripts

The trained models already exist for the dissertation results. The following
commands are examples only and should not be run unless retraining is intended.

```powershell
python train_knn_hog.py `
  --train_csv outputs\dataset\train_balanced.csv `
  --val_csv outputs\dataset\val_balanced.csv `
  --test_csv outputs\dataset\test_balanced.csv `
  --output_dir outputs
```

```powershell
python train_linear_svm_hog.py `
  --train_csv outputs\dataset\train_balanced.csv `
  --val_csv outputs\dataset\val_balanced.csv `
  --test_csv outputs\dataset\test_balanced.csv `
  --output_dir outputs
```

```powershell
python train_cnn_gradcam.py `
  --train_csv outputs\dataset\train_balanced.csv `
  --val_csv outputs\dataset\val_balanced.csv `
  --test_csv outputs\dataset\test_balanced.csv `
  --output_dir outputs\cnn
```

```powershell
python train_vit.py --output_dir outputs\vit
```

## Explainability Analysis

The cross-model explainability script loads the saved trained models and does
not retrain them. It applies occlusion sensitivity in image coordinates.

For HOG-based models, each occluded image passes through:

```text
image preprocessing -> HOG extraction -> fitted StandardScaler -> trained classifier
```

For CNN and ViT, the evaluation-time resizing and normalisation pipelines are
used.

To generate representative qualitative examples grouped by true label:

```powershell
python explainability_comparison.py `
  --qualitative_examples `
  --qualitative_per_label 3 `
  --patch_size 32 `
  --stride 32 `
  --output_dir outputs\qualitative_comparison
```

This produces:

```text
outputs/qualitative_comparison/qualitative/original_cross_model_heatmaps.png
outputs/qualitative_comparison/qualitative/manipulated_cross_model_heatmaps.png
outputs/qualitative_comparison/qualitative/qualitative_cross_model_results.csv
```

The qualitative figures are representative examples only. They should not be
used to report new accuracy estimates.

## Interpreting Heatmaps

The heatmaps show model sensitivity to spatial occlusion, not ground-truth
manipulation masks.

In the qualitative comparison figures, each heatmap is oriented towards the
class predicted by that model:

- If a model predicts **Manipulated**, warmer colours indicate regions whose
  occlusion reduces evidence for the Manipulated decision.
- If a model predicts **Real**, warmer colours indicate regions whose occlusion
  reduces evidence for the Real decision.

The displayed heatmaps are normalised separately for visual inspection, while
the raw arrays are saved for reference. Therefore, colour intensity should not be
compared directly across models as an absolute measure of explanation strength.

For KNN, probability values are based on discrete nearest-neighbour votes. With
`k=11`, probabilities can only take values such as `0/11`, `1/11`, ..., `11/11`.
Some KNN heatmaps may therefore appear weak when local occlusions do not change
the neighbour vote.

## Local Forensic Demo Tool

I uploaded the model to Hugging Face, and deployed it on Streamlit, you can click the website to view it
https://deepfake-forensics.streamlit.app/


## Files Usually Excluded From GitHub

The following files/folders are generated outputs, datasets or large model
artifacts and are normally excluded from a code-review repository:

```text
outputs/
Celeb-DF-v2/
ffhq/
stargan/
stylegan_ffhq/
pggan_v1/
pggan_v2/
faceapp/
forensic_tool/static/results/
forensic_tool/*.log
forensic_tool/upload_test_response.html
__pycache__/
.skimage_cache/
```

If model checkpoints are required for reproduction, use Git LFS or provide an
external download link and place the files in the expected `outputs/` paths.

## Notes for Reviewers

- The reported dissertation results are based on the saved outputs generated
  from the independent test split.
- The explainability figures are qualitative and should be interpreted alongside
  the quantitative performance metrics and confusion matrices.
- The local forensic tool is intended as a demonstration of the trained ViT
  model, not as a production forensic system.
