# SignMate

A small web app for Indian Sign Language (ISL). You can:

- sign letters in front of the webcam and get text
- speak into the mic and see ISL fingerspelling
- type a phrase and play the signs as a slideshow

The folder is named `SignSync`; the app itself is called SignMate.

Most of this runs in the browser. The Python backend handles login, history, and the letter-recognition models.

---

## What it does

**Sign → Text**  
Webcam + MediaPipe hand landmarks + a few Keras models. Single-hand letters are `C I L O U V`; the rest of A–Z are two-handed ISL. Extra models try to unstick C/K and J/K/S/Y, which look similar.

You can add letters by hand, or turn on auto-add when a sign is held still. Space, backspace, copy, and speak-aloud are in the same panel.

**Speech → ISL**  
Uses the browser’s speech recognition (Chrome / Edge). The sentence is broken into fingerspelling. The slideshow plays once and stops; Replay starts it again. There is also a strip view of every sign.

**Text → ISL**  
Same as speech, but you type (or click a sample chip like HELLO / INDIA).

**Dictionary**  
Photos for A–Z, with one-hand vs two-hand filters. Click a letter for a larger picture and a short how-to.

**History**  
Saved translations (sign / speech / text). Filter, search, export JSON or TXT. Logged-in users go to SQLite; guests stay in local storage.

**Dataset helpers** (optional)  
`tools/record_word_dataset.py` records landmark sequences. `tools/train_word_model.py` trains a word classifier on those files. Word list notes live in `dataset/words/` and point at INCLUDE and the ISLRTC dictionary.

---

## Stack

- Backend: Python 3.10+, FastAPI, Uvicorn, SQLAlchemy, SQLite
- ML: MediaPipe, TensorFlow/Keras, OpenCV, NumPy
- Frontend: HTML / CSS / JS (no framework)
- Browser APIs: camera, Speech Recognition, Speech Synthesis

---

## How to run

Python 3.10+ and Chrome or Edge (speech needs them).

1. Open this folder in VS Code / Cursor.
2. Press F5 (`SignMate: Launch Web App`) **or** run:

```cmd
.\run.bat
```

```powershell
.\run.ps1
```

3. Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

If the venv is missing, create it and `pip install -r requirements.txt` first.

---

## Folder layout

```
SignSync/
├── .vscode/           debug / tasks
├── app/
│   ├── auth.py        register / login
│   ├── database.py
│   ├── main.py        routes + dictionary API
│   ├── ml_ck.py       C/K model
│   ├── ml_combined.py main letter pipeline
│   ├── ml_jksy.py     J/K/S/Y model
│   ├── ml_service.py  RealSign + MediaPipe
│   └── models.py      User, TranslationHistory
├── dataset/words/     word list, landmarks, citations
├── frontend/          html, css, js, letter photos
├── tools/             record / train scripts
├── run.bat
├── run.ps1
├── requirements.txt
└── README.md
```

---

## Citations

If you use the word dataset notes in a paper, cite INCLUDE:

```bibtex
@inproceedings{include2020,
  author    = {Sridhar, Prem and Ganesan, Ramesh and others},
  title     = {INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition},
  booktitle = {Proceedings of the 28th ACM International Conference on Multimedia (MM '20)},
  year      = {2020},
  publisher = {ACM}
}
```
