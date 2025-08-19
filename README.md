# 🐾 PetVision – Image Classification (Dog, Cat or Other)

## 📌 Descrizione
PetVision è un progetto di machine learning che permette di classificare un’immagine come:
- **Cane 🐶**
- **Gatto 🐱**
- **Altro 🖼️**

L’obiettivo è dimostrare, passo dopo passo, come costruire un modello di **image classification** documentando:
1. Creazione e preprocessing del dataset
2. Addestramento del modello (CNN con TensorFlow/Keras o PyTorch)
3. Valutazione delle performance
4. Deployment in **Docker** per garantire portabilità e facilità di utilizzo

---

## 📂 Struttura del progetto
```
PetVision/
│── data/                 # Dataset (raw + processed)
│── notebooks/            # Jupyter notebooks per esplorazione e training
│── models/               # Modelli addestrati salvati
│── src/                  # Codice sorgente (preprocessing, training, inference)
│── app/                  # Eventuale interfaccia web/app
│── requirements.txt      # Dipendenze Python
│── Dockerfile            # Configurazione Docker
│── README.md             # Documentazione principale
```

---

## 🚀 Getting Started

### 1️⃣ Clona il repository
```bash
git clone https://github.com/tuo-username/PetVision.git
cd PetVision
```

### 2️⃣ Crea un ambiente virtuale (opzionale se non usi Docker)
```bash
python -m venv venv
source venv/bin/activate   # su Linux/Mac
venv\Scripts\activate      # su Windows
```

### 3️⃣ Installa le dipendenze
```bash
pip install -r requirements.txt
```

---

## 🐳 Esecuzione con Docker

### Build dell’immagine
```bash
docker build -t petvision .
```

### Avvio del container
```bash
docker run -it --rm -p 8501:8501 petvision
```

Se hai incluso una webapp (es. **Streamlit**), sarà accessibile su:
👉 http://localhost:8501

---

## 🧑‍💻 Utilizzo

### Addestramento del modello
```bash
python src/train.py --epochs 20 --batch_size 32
```

### Predizione su un’immagine
```bash
python src/predict.py --image path/to/immagine.jpg
```

---

## 📊 Risultati
Il modello raggiunge un’accuratezza di circa **XX%** sul validation set.  
Esempi di predizioni:  
- `dog.jpg` → **Cane**
- `cat.png` → **Gatto**
- `car.jpg` → **Altro**

---

## 🛠️ Tecnologie utilizzate
- Python 3.x
- TensorFlow/Keras (o PyTorch)
- NumPy, Pandas, Matplotlib
- OpenCV / Pillow
- Docker
- (Opzionale) Streamlit/Flask per la webapp

---

## 📌 To-Do
- [ ] Rifinire preprocessing immagini  
- [ ] Hyperparameter tuning  
- [ ] Aggiungere data augmentation  
- [ ] Creare una webapp demo  

---

## 📜 Licenza
Distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori dettagli.
