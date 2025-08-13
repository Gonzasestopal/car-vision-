# 🚗 Kopiloto Vision

Kopiloto Vision is a prototype AI-powered assistant that analyzes car images to detect visible damage and estimate resale value.
This project was built as a quick MVP for demonstrating how AI can accelerate car inspections in the automotive industry.

---

## 📌 Features
- **Upload multiple car images** (JPEG, PNG)
- **Preview uploaded images** in the browser
- **AI analysis placeholder** for:
  - Damage detection (scratches, dents, missing parts)
  - Price estimation
- **Streamlit interface** for quick prototyping

---

## 🛠️ Tech Stack
- [Python 3.8+](https://www.python.org/)
- [Streamlit](https://streamlit.io/) – Interactive UI
- [Pillow](https://pillow.readthedocs.io/) – Image processing
- *(Optional)* [YOLOv8](https://github.com/ultralytics/ultralytics) – Object detection model
- *(Optional)* [scikit-learn](https://scikit-learn.org/) – Price prediction model

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/kopiloto-vision.git
cd kopiloto-vision
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at:

http://localhost:8501
