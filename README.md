# DR Classifier — Local Web App

A Streamlit demo that loads your trained checkpoint and lets you upload a
fundus image to see the predicted DR grade plus a Grad-CAM explanation.

## Setup

1. Download `best_model.pt` from your Colab run (Cell 22's `CHECKPOINT_PATH`,
   or your Drive backup) and place it in this same folder.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```
   This opens automatically in your browser, typically at `http://localhost:8501`.

If your checkpoint file is named differently or lives elsewhere, set an
environment variable instead of moving it:
```bash
APTOS_CHECKPOINT=/path/to/best_model.pt streamlit run app.py
```

## What it does

- Applies the exact same preprocessing as training (circular crop, illumination
  normalization, CLAHE) so predictions match what you saw in Cell 21.
- Runs the image through the ResNet101 ordinal-regression model.
- Converts the raw score to a class using the thresholds optimized on your
  validation set (also loaded from the checkpoint).
- Shows a Grad-CAM heatmap so you can see what the model is attending to.

## Sanity check before showing it to anyone

Upload a few images from your test split that you already have ground-truth
labels for, and confirm the app's prediction matches what Cell 21 reported for
those same images. If it doesn't, the most likely cause is a preprocessing or
image-size mismatch between this app and the notebook — check `IMG_SIZE` in
the sidebar matches what you trained with.
