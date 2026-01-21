# ♻️ Smart Waste Sorter: AI-Powered Biodegradability System

This repository contains a complete **Smart Waste Sorter system** built with a Streamlit dashboard for inference and YOLOv8 for real-time object detection. The system is designed to classify waste into 10 categories and automatically determine if an item is **Biodegradable** or **Non-Biodegradable**.

---

## 🔧 Features

- 📟 **Streamlit Dashboard** for real-time inference (Upload, Snapshot, & Live Stream) 
- 📊 **Dynamic Analytics** with Plotly charts and SQLite database tracking  
- 🍃 **Biodegradability Logic** mapped specifically for 10 waste categories 
- ⚡ **GPU Accelerated workflow** optimized for NVIDIA CUDA  
- ⚡ **Fast local inference workflow**

---

## 📦 Dataset

This project uses the **Garbage Classification V2 Dataset** created by **Sumn2u**.

📌 **Dataset Link:**  
https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2

---

## 🖥️ Running the Application (Dashboard)

Navigate into the dashboard folder:

```bash
cd Smart-Waste-Sorter-YOLOv8
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the development server:
```bash
streamlit run app/app.py
```


🧠 Model Training (Notebooks)

All training-related code is in the notebooks/ directory. It includes:

🧪 Evaluation & preprocessing modules
🔍 Trash Detection training notebook

You may use any Python environment or Google Colab to run them.

📁 Project Structure
```bash
.
├── app/                # Streamlit UI & SQLite database
├── notebooks/          # training notebooks
├── outputs/            # Saved models (best.pt) and training logs
├── reports/            # Final Project Report & PPT Presentation 
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── ...
```


⚖️ License & Attribution Notice

This project may include or reference external datasets and libraries. 
The Garbage Classification V2 Dataset belongs to its creator Sumn2u, and attribution is mandatory when using it.

Please comply with any dataset licensing rules stated on Kaggle.

⭐ Acknowledgements

Sumn2u — creator of the dataset

Contributors and collaborators

n-Alan777

