# CropInsight

CropInsight demonstrates complete CNN and LSTM pipelines for crop analysis.
The training scripts automatically use a downloaded sign language digits
dataset when available and fall back to the built-in digits dataset from
`scikit-learn` so the example works offline.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train the CNN and LSTM models:

```bash
python train_cnn.py
python train_lstm.py
```

## Prediction

After training, run predictions (optionally supply an image path):

```bash
python predict.py --image path/to/image.jpg
```
