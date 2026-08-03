const fs = require("fs")
const path = require("path")

const root = path.resolve(__dirname, "..")
const modelPath = path.join(root, "public", "model", "financial_router.json")
const indexPath = path.join(root, "public", "index.html")

if (!fs.existsSync(modelPath)) {
  throw new Error("Missing public/model/financial_router.json. Run python scripts/train_keras_financial_router.py first.")
}

if (!fs.existsSync(indexPath)) {
  throw new Error("Missing public/index.html.")
}

console.log("Static Keras financial extractor is ready.")
