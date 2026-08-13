# Media Manipulation Forensics Tool

This is a local demo tool built around the best-performing ViT checkpoint in
`outputs/vit/models/vit_best_model`.

It supports:

- image upload
- video upload
- Manipulated probability from the saved ViT model
- ViT attention-rollout overlay for suspected manipulated images/frames
- red-box visual highlighting for high-attention regions
- per-frame CSV report for videos

The tool does not retrain or modify the saved model.

## Run

```powershell
python forensic_tool/app.py
```

Open:

```text
http://127.0.0.1:7860
```

To use a different port:

```powershell
$env:FORENSIC_TOOL_PORT="7861"; python forensic_tool/app.py
```

## Video Settings

The upload page exposes:

- video sample interval, default `1.0` second
- maximum sampled frames, default `32`

For long videos, increasing the interval or lowering maximum sampled frames makes
the tool faster.

## Interpretation

The score is the ViT model's probability for the Manipulated class. The overlay
is an attention-based visualisation, not proof of the exact edited pixels. It
shows the regions that the model focused on when making its prediction, which is
useful for triage and digital-forensics explanation.
