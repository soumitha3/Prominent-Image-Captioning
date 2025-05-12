import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import load
from googletrans import Translator
from gtts import gTTS
import os
from playsound import playsound

# Load essentials
tokenizer = load(open("tokenizer.p", "rb"))
model = load_model("models/model_9.h5")
xception_model = Xception(include_top=False, pooling="avg")
max_length = 32
translator = Translator()

# Globals
selected_image_path = None
photo_feature = None
selected_lang_code = "en"
current_caption = ""

# Language options
language_map = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Telugu": "te",
    "Tamil": "ta",
    "Bengali": "bn",
    "Kannada": "kn"
}

def extract_features(filename, model):
    try:
        image = Image.open(filename).convert('RGB')
        image = image.resize((299, 299))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 127.5 - 1.0
        return model.predict(image)
    except:
        messagebox.showerror("Error", "Could not process the image.")
        return None

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None or word == 'end':
            break
        in_text += ' ' + word
    return in_text.replace('start', '').strip()

def upload_image():
    global selected_image_path, photo_feature
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    selected_image_path = file_path
    photo_feature = extract_features(file_path, xception_model)
    if photo_feature is None:
        return

    img = Image.open(file_path).resize((480, 360))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    caption_label.config(text="")
    generate_btn.config(state="normal")
    play_btn.config(state="disabled")

def generate_caption():
    global current_caption
    if photo_feature is None:
        messagebox.showerror("Error", "No image loaded.")
        return

    description = generate_desc(model, tokenizer, photo_feature, max_length)

    try:
        translated = translator.translate(description, dest=selected_lang_code).text
    except Exception as e:
        translated = description
        messagebox.showwarning("Translation failed", f"Error: {e}")

    current_caption = translated
    caption_label.config(text=f"Caption:\n{translated}")
    play_btn.config(state="normal")

def play_caption():
    global current_caption
    if not current_caption:
        return
    try:
        tts = gTTS(text=current_caption, lang=selected_lang_code)
        tts.save("temp_caption.mp3")
        playsound("temp_caption.mp3")
        os.remove("temp_caption.mp3")
    except Exception as e:
        messagebox.showwarning("Audio Error", f"Could not play the caption.\n{e}")

def change_language(lang):
    global selected_lang_code
    selected_lang_code = language_map[lang]

# --- UI Setup ---
root = tk.Tk()
root.title("Image Caption Generator with Voice")
root.geometry("900x780")
root.configure(bg="#f5f5f5")

title = tk.Label(root, text="Automated Captioning of Images", font=("Helvetica", 22, "bold"), bg="#f5f5f5", fg="#333")
title.pack(pady=10)

# Language Selector
lang_frame = tk.Frame(root, bg="#f5f5f5")
lang_frame.pack(pady=5)

lang_label = tk.Label(lang_frame, text="Language:", font=("Arial", 12), bg="#f5f5f5")
lang_label.pack(side=tk.LEFT, padx=5)

lang_option = tk.StringVar(value="English")
lang_menu = tk.OptionMenu(lang_frame, lang_option, *language_map.keys(), command=change_language)
lang_menu.config(font=("Arial", 12))
lang_menu.pack(side=tk.LEFT)

# Image & Buttons
btn_frame = tk.Frame(root, bg="#f5f5f5")
btn_frame.pack(pady=10)

upload_btn = tk.Button(btn_frame, text="📁 Upload Image", command=upload_image, font=("Arial", 13), bg="#2196F3", fg="white", padx=10)
upload_btn.pack(side=tk.LEFT, padx=10)

generate_btn = tk.Button(btn_frame, text="🧠 Generate Caption", command=generate_caption, font=("Arial", 13), bg="#4CAF50", fg="white", padx=10, state="disabled")
generate_btn.pack(side=tk.LEFT, padx=10)

play_btn = tk.Button(btn_frame, text="🔊 Play Caption", command=play_caption, font=("Arial", 13), bg="#FF9800", fg="white", padx=10, state="disabled")
play_btn.pack(side=tk.LEFT, padx=10)

# Image Display
image_label = tk.Label(root, bg="#f5f5f5")
image_label.pack(pady=20)

# Caption Display
caption_label = tk.Label(root, text="Caption will appear here", font=("Arial", 15), bg="#f5f5f5", wraplength=750, justify="center")
caption_label.pack(pady=20)

root.mainloop()
