# Keras Financial Document Extractor

Real Keras-backed financial document extraction demo.

## Train

```bash
python scripts/train_keras_financial_router.py
```

The training script creates a labeled financial-document routing corpus, trains a TensorFlow/Keras classifier, saves the `.keras` model, exports browser-readable weights to `public/model/financial_router.json`, and writes metrics to `reports/`.

## Run

```bash
npm install
npm run dev
```

The web app accepts pasted text plus uploaded TXT, CSV, JSON, and PDF files. It extracts common financial fields, preserves raw text, and flags unknown or low-confidence inputs.
