# Public Streamlit deployment for the forensic tool

This folder contains a Streamlit version of the ViT-based forensic demo.

## Why this version

The original `forensic_tool/app.py` is a custom local HTTP server and saves
uploaded media and generated results under `static/results`. The Streamlit
version avoids a permanent results directory:

- images are processed in memory;
- videos are written only to an OS temporary file for OpenCV frame extraction;
- the temporary video is deleted in a `finally` block immediately after analysis;
- heatmaps and CSV output are generated in memory;
- no upload history is implemented.

Important: on a public web deployment, a user's file still has to be transmitted
from the browser to the hosting server for inference. This version is designed
for **no intentional application-level retention**, not "the file never leaves
the user's device". Strict device-only processing would require a browser-side
model (for example ONNX/WebGPU) or continued local use.

## Local test

From the repository root:

```bash
pip install -r forensic_tool/requirements_streamlit.txt
streamlit run forensic_tool/streamlit_app.py
```

By default the app expects the checkpoint at:

```text
outputs/vit/models/vit_best_model
```

You can override it:

```bash
export FORENSIC_MODEL_SOURCE=/path/to/vit_best_model
streamlit run forensic_tool/streamlit_app.py
```

## Streamlit Community Cloud

The current GitHub repository does not contain
`outputs/vit/models/vit_best_model`, so the checkpoint must be made available
before the public app can load.

Recommended approach for a dissertation demo:

1. Upload the saved Hugging Face-format `vit_best_model` folder to a model
   repository on Hugging Face Hub.
2. In Streamlit Community Cloud, deploy this GitHub repository.
3. Set the main file path to:

```text
forensic_tool/streamlit_app.py
```

4. Add a Streamlit secret:

```toml
MODEL_SOURCE = "your-huggingface-username/your-model-repository"
```

5. Deploy and share the generated Streamlit URL with supervisors.

If the model repository is private, additional authentication handling is
required. For a short-lived dissertation demonstration, a public model
repository is simplest, provided you are comfortable publishing the checkpoint.

## Interpretation

This is a research screening prototype. A `Real` or `Manipulated` prediction is
not a definitive forensic conclusion. Attention rollout is an exploratory
explanation of model behaviour, not ground-truth localisation of edited pixels.
