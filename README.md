# AI-Generated Facial Media Detection

This repository contains the code developed for an MSc dissertation on the
detection of authentic and manipulated facial media.

The project compares four classification approaches:

- KNN + HOG
- Linear SVM + HOG
- CNN
- Vision Transformer (ViT)

The task is formulated as binary classification:

- `0` = Real
- `1` = Manipulated

## Detection Scope

The current system is mainly designed for **AI-generated and deep-generative
facial manipulation** represented in the training datasets, including:

- PGGAN-generated faces
- StyleGAN-generated faces
- StarGAN facial editing
- FaceApp-style facial manipulation
- Celeb-DF deepfake facial synthesis

The detector is therefore intended as a **face-centred manipulation screening
tool**, rather than a general-purpose image forensics system.

It has not been specifically trained or validated for traditional manipulation
types such as copy-move forgery, splicing, object insertion/removal or
inpainting. These could be incorporated in future work using additional
datasets and model training.

## Repository Structure

```text
.
├── Photos-Videos-Manipulations-Dataset/
├── forensic_tool/
├── prepare_dataset.py
├── create_balanced_splits.py
├── train_knn_hog.py
├── train_linear_svm_hog.py
├── train_svm_hog.py
├── train_cnn_gradcam.py
├── train_vit.py
├── explainability_comparison.py
├── requirements.txt
└── README.md
````

## Main Scripts

| File                           | Purpose                                                      |
| ------------------------------ | ------------------------------------------------------------ |
| `prepare_dataset.py`           | Prepares images and video frames and creates dataset splits. |
| `create_balanced_splits.py`    | Balances the train, validation and test partitions.          |
| `train_knn_hog.py`             | Trains and evaluates KNN using HOG features.                 |
| `train_linear_svm_hog.py`      | Trains and evaluates the linear SVM using HOG features.      |
| `train_cnn_gradcam.py`         | Trains the CNN and generates Grad-CAM explanations.          |
| `train_vit.py`                 | Fine-tunes and evaluates the ViT model.                      |
| `explainability_comparison.py` | Generates cross-model occlusion sensitivity heatmaps.        |
| `forensic_tool/`               | Contains the local and Streamlit forensic screening tools.   |

## Explainability

The project compares spatial model behaviour using explainability techniques.

The generated heatmaps show which image regions influence model predictions.
They should be interpreted as exploratory explanations of model behaviour, not
as ground-truth localisation of manipulation artefacts.

## Online Demo

A ViT-based prototype has been deployed using Streamlit:

**[https://deepfake-forensics.streamlit.app/](https://deepfake-forensics.streamlit.app/)**

The application supports image and video upload and returns a `Real` or
`Manipulated` prediction together with model confidence and visual explanation.

For videos, sampled frames are analysed individually using the trained
image-based ViT detector.

The tool is intended for research demonstration and forensic screening only. It
should not be treated as a definitive determination of media authenticity.

## Model Availability

The trained ViT model used by the online demo is hosted on Hugging Face:

```text
Penny1507288/deepfake_detection
```

Most trained checkpoints, datasets and generated experiment outputs are not
included in this GitHub repository because of file size.

The repository mainly contains the code required to reproduce the experimental
pipeline and explainability analysis.

## Future Work

Future extensions could include:

* copy-move forgery detection
* image splicing
* object insertion/removal
* inpainting
* diffusion-generated imagery
* temporal deepfake video analysis
* broader cross-dataset robustness testing

## Disclaimer

This repository contains a research prototype developed for academic purposes.
Model outputs should support, rather than replace, formal forensic examination.
