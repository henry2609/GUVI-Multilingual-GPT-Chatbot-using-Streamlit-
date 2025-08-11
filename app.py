import os
from huggingface_hub import login
login(os.environ["HF_TOKEN"])  # Secure token from HF secrets

import gradio as gr
import faiss
import numpy as np
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# === Load Lightweight Model (Demo-Friendly) ===
model_name = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# === Translator (NLLB) ===
nllb_model = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model)
nllb = AutoModelForSeq2SeqLM.from_pretrained(nllb_model)
translator = pipeline("translation", model=nllb, tokenizer=nllb_tokenizer)

# === Language codes ===
lang_code_map = {
    "ta": "tam_Taml", "hi": "hin_Deva", "te": "tel_Telu", "ml": "mal_Mlym", "kn": "kan_Knda",
    "bn": "ben_Beng", "mr": "mar_Deva", "fr": "fra_Latn", "de": "deu_Latn", "ko": "kor_Hang",
    "zh": "zho_Hans", "zh-cn": "zho_Hans", "ja": "jpn_Jpan", "en": "eng_Latn"
}

# === FAISS + Embeddings ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
try:
    index = faiss.read_index("guvi_faiss.index")
    with open("chunks.txt", "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f.readlines()]
except Exception as e:
    print("⚠️ Index/chunks load failed:", e)
    index = None
    chunks = []

# === Prompt Builder ===
def build_rag_prompt(context_chunks, question):
    context_text = "\n".join(context_chunks)
    return f"""You are a multilingual chatbot for GUVI. The user may ask questions in any language. Answer in English, and only based on the context given.

### Context:
{context_text}

### Question:
{question}

### Answer:"""

# === Chatbot Logic ===
def guvi_chatbot(user_input, history, user_lang):
    if not chunks or not index:
        return history + [[user_input, "⚠️ System not ready. Try again later."]]

    try:
        lang = detect(user_input)
        src_lang = lang_code_map.get(lang, "eng_Latn")
        if not user_lang:
            user_lang = src_lang

        original_input = user_input  # Preserve for display

        if lang != "en":
            try:
                translated_input = translator(user_input, src_lang=src_lang, tgt_lang="eng_Latn")[0]['translation_text']
            except:
                return history + [[original_input, "⚠️ Translation failed. Try in English."]]
        else:
            translated_input = user_input

        q_embed = embedder.encode([translated_input])
        D, I = index.search(np.array(q_embed), 3)
        context_chunks = [chunks[i] for i in I[0]]

        prompt = build_rag_prompt(context_chunks, translated_input)
        try:
            result = generator(prompt, max_new_tokens=200)[0]["generated_text"]
            answer = result.strip()
        except:
            answer = "⚠️ Sorry, generation failed."

        if user_lang != "eng_Latn":
            try:
                answer = translator(answer, src_lang="eng_Latn", tgt_lang=user_lang)[0]['translation_text']
            except:
                answer += "\n(⚠️ Back translation failed.)"

        history.append([original_input, answer])
        return history

    except Exception as e:
        return history + [[user_input, "⚠️ Unexpected error. Try again."]]

# === Gradio UI ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤖 GUVI Multilingual Chatbot")
    gr.Markdown("Ask me anything about GUVI in any language!")

    chatbot = gr.Chatbot(value=[["🤖", "Hi! How can I help you today?"]])
    user_input = gr.Textbox(label="Ask me something...")
    state = gr.State([])
    user_lang = gr.State(None)

    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear")

    send_btn.click(fn=guvi_chatbot, inputs=[user_input, state, user_lang], outputs=chatbot)
    send_btn.click(fn=lambda x: "", inputs=user_input, outputs=user_input)
    clear_btn.click(lambda: [["🤖", "Hi! How can I help you today?"]], outputs=chatbot)
    clear_btn.click(lambda: [], outputs=state)

demo.launch()
