# **🧠 Fake News & Claim Veracity Explainer**

### **Project Overview**

This project is an **interactive web app** that allows users to enter political or public claims and receive feedback on how likely they are to be **false, true, or somewhere in between** — based on pretrained models and explainability techniques like SHAP or LIME.

The project’s purpose is twofold:

1. Help users understand the _credibility_ of a statement.  

2. Provide an _interpretable explanation_ for why a model arrived at that judgment.  

## **✅ What We’ve Done So Far**

### **🔧 Data Handling**

- Used a dataset (e.g., LIAR) consisting of claims, metadata, and labeled truth values (true, false, half-true, etc.).  

- Preprocessed the dataset:  
  - Combined fields into a single text input.  

  - Encoded class labels using LabelEncoder.  

### **🤖 Modeling**

- **Initial Baseline**: Trained a LogisticRegression model on TF-IDF vectors to validate pipeline and explanation.  

- **Explainability**: Used **SHAP** for logistic regression explanations (feature importance of words).  

- **Transformer Model**: Trained a DistilBERT classifier using HuggingFace Transformers.  
  - Tokenized input claims.  

  - Evaluated performance on validation data.  

  - Saved model and tokenizer.  

### **🧪 Explainability (Current)**

- SHAP was **initially integrated** for DistilBERT but resulted in \[MASK\] artifacts due to tokenizer behavior. This is paused for now.  

- Instead, we return a basic prediction with a class confidence for now.  

### **🌐 Frontend (React + Next.js + Tailwind CSS)**

- User can enter a claim.  

- Sends query to the backend for prediction.  

- Renders result and explanation nicely.  

- Fixed layout, dark mode styling, and spacing for usability.  

### **🛠 Backend (FastAPI)**

- Exposes /explain endpoint for predictions.  

- Currently returns:  
  - Predicted class label.  

  - Confidence score.  

- CORS-enabled for frontend communication.  

## **💡 Key Design Decisions**

| **Decision** | **Rationale** |
| --- | --- |
| **Start with DistilBERT** | Faster to train and test. Smaller model footprint. |
| --- | --- |
| **Use SHAP over LIME (initially)** | SHAP offers theoretically grounded explanations; however, limitations exist with tokenizer. |
| --- | --- |
| **FastAPI for backend** | Lightweight, async-friendly, easy ML model integration. |
| --- | --- |
| **Next.js + Tailwind** | Modern, quick to iterate frontend with built-in SSR and simple styling. |
| --- | --- |
| **Return simplified result for now** | Prioritize user experience while debugging explainer logic. |
| --- | --- |

## **🔭 What More Can Be Done**

### **Short-Term**

- ✅ ✅ **Fix SHAP mask bug** with BERT-compatible Tokenizer objects or switch to LIMETextExplainer instead.  

- 🧪 Compare model explanations using SHAP vs LIME side by side.  

- 🔁 Evaluate other models like RoBERTa or BART with explanations.  

- 📊 Add proper evaluation + metrics dashboard.  

- 🎯 Color-code class labels visually in frontend.  

### **Mid-Term**

- 🔍 Show similar examples from training data (case-based reasoning).  

- 🧠 Add "confidence interval" explanation alongside predictions.  

- 🌐 Add multilingual support.  

- 📄 Let users upload articles to evaluate batches of claims.  

### **Long-Term / What This Could Morph Into**

- **Journalism Plugin / API**: Embeddable tool for fact-checkers and newsrooms.  

- **Browser Extension**: Real-time highlight of claims with explanations.  

- **Trust-Scoring System**: Aggregated scores over time for public figures or sources.  

- **Educational Tool**: For classrooms teaching bias, media literacy, and propaganda.  

- **Enterprise Use**: Monitor political statements, corporate press releases, etc.  

## **📌 Next Steps**

1. Fix tokenization or use LIME on current transformer.  

2. Rebuild SHAP pipeline around simpler tokenization.  

3. Extend backend to handle batch inputs.  

4. Improve frontend UX (color, history of queries).  

5. Expand dataset / consider fine-tuning with newer data.