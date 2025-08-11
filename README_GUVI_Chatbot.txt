
# 🤖 GUVI Multilingual Chatbot (Streamlit + Colab + ngrok)

A **multilingual AI chatbot** for GUVI that understands and answers user queries in **Tamil, Hindi, Telugu, Malayalam, and English**.  
It uses **semantic search**, **knowledge base retrieval**, and **text generation** to provide accurate answers — always responding in the **same language** the question was asked.

---

## 🚀 Features

- **Multilingual**: Tamil, Hindi, Telugu, Malayalam, English.  
- **Knowledge Base Search**: Finds the most relevant answer from your Q/A dataset.  
- **Semantic Search**: Uses `sentence-transformers` to match questions even if wording changes.  
- **Optional Text Generation**: Uses `flan-t5` to create answers when KB confidence is low.  
- **Automatic Translation**: Uses Facebook NLLB-200 for high-quality translations.  
- **Local Language Responses**: Bot replies in the language of the question.  
- **Colab + Streamlit UI**: Run fully in Google Colab with ngrok public URL.  
- **Logging**: Saves each query and response for later review.  

---

## 📂 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Required Python packages
├── guvi_qa_table.xlsx      # Knowledge Base (Q/A pairs)
├── guvi_chat_logs.csv      # Auto-generated logs of user queries
└── README.md               # Project documentation
```

---

## 🛠️ Installation & Running in Google Colab

### 1️⃣ Open Colab
- Go to [Google Colab](https://colab.research.google.com/).

### 2️⃣ Install dependencies
```python
!pip install -q streamlit transformers sentence-transformers langdetect pyngrok pandas openpyxl python-docx datasets accelerate
```

### 3️⃣ Upload files
- `guvi_qa_table.xlsx` — your knowledge base file.
- (Optional) A cleaned/tokenized version.

### 4️⃣ Set your ngrok token
```python
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"
```
You can get a free token from: [ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)

### 5️⃣ Run the app
```python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("🚀 Public URL:", public_url)

!streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 🧠 How It Works

1. **Language Detection**  
   Detects the input language using `langdetect`.

2. **Translation to English**  
   Uses **Facebook NLLB-200** to translate the query to English for searching.

3. **Semantic Search in KB**  
   Encodes all KB questions using `paraphrase-multilingual-MiniLM-L12-v2` and finds the closest match.

4. **Answer Retrieval / Generation**  
   - If match score is high → returns KB answer.  
   - If match score is low → uses `flan-t5-small` to generate an answer using KB context.

5. **Translation Back**  
   Translates the English answer back into the detected local language.

6. **UI Display**  
   Shows the answer in local language.

---

## 📸 Screenshots

| Tamil Query | English Query |
|-------------|---------------|
| ![Tamil Chatbot Example](screenshots/tamil_query.png) | ![English Chatbot Example](screenshots/english_query.png) |

> Place your screenshots in a folder named `screenshots/` inside the project.  

---

## 📊 Example Interaction

**User (Tamil)**:  
```
GUVI-வில் என்ன வகையான படிப்புகள் உள்ளன?
```

**Bot (Tamil)**:  
```
Blogs, PDFs, handbooks, interview tips.
```

---

## 📌 Requirements

- Python 3.8+  
- Streamlit  
- Transformers  
- Sentence-Transformers  
- langdetect  
- Facebook NLLB-200 model  
- flan-t5-small (or your fine-tuned generator model)  
- Ngrok account for public URL (optional)  

---

## 🔮 Future Improvements

- Add **speech-to-text** for voice input.  
- Deploy on **Streamlit Cloud** or **Hugging Face Spaces**.  
- Enhance **UI with chatbot bubbles** and avatars.  
- Enable **contextual multi-turn conversation**.  
