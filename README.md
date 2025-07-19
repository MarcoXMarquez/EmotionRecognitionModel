# 🎭 Emotion Recognition Model

Proyecto de reconocimiento de emociones faciales usando una CNN personalizada en **PyTorch**, entrenado con un dataset balanceado en `data_combined/`. Soporta entrenamiento, evaluación e inferencia sobre imágenes faciales.

---

## 🧭 Tabla de Contenidos

- [🔧 Requisitos](#requisitos)
- [📁 Estructura del Proyecto](#estructura-del-proyecto)
- [🧠 Arquitectura del Modelo](#modelo)
- [📊 Entrenamiento](#entrenamiento)
- [🧪 Evaluación](#evaluación)
- [🔍 Inferencia](#inferencia)
- [🖼️ Visualización](#visualización)
- [📦 Distribución](#distribución)
- [📜 Licencia](#licencia)
- [📬 Autor](#autor)

---

## 🔧 Requisitos
---

## 📦 Requerimientos y Librerías Principales

| Paquete                  | Versión         | Descripción breve                                                |
|--------------------------|-----------------|-------------------------------------------------------------------|
| [Python](https://www.python.org/)          | 3.10+           | Lenguaje de programación principal usado en el proyecto.         |
| [torch](https://pytorch.org/)             | 2.7.1+cu118     | Framework de deep learning para entrenamiento e inferencia.      |
| [torchvision](https://pytorch.org/vision/) | 0.22.1          | Conjuntos de datos, modelos y transformaciones para visión.      |
| [torchaudio](https://pytorch.org/audio/)   | 2.7.1+cu118     | Librería para trabajar con datos de audio (usada por compatibilidad). |
| [opencv-python](https://pypi.org/project/opencv-python/) | 4.12.0.88      | Procesamiento de imágenes, usado para detección de rostros.      |
| [scikit-learn](https://scikit-learn.org/)  | 1.7.0           | Herramientas para evaluación y métricas de ML.                   |
| [numpy](https://numpy.org/)               | 2.2.6           | Manipulación eficiente de tensores y datos numéricos.            |
| [matplotlib](https://matplotlib.org/) (opcional) | -           | Visualización de datos (si se integra en el viewer).             |
| [pyinstaller](https://www.pyinstaller.org/) | 6.14.2          | Empaquetado del modelo como ejecutable.                          |

Instalación rápida de PyTorch (CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Luego instala todo con:

```bash
pip install -r requirements.txt
```


Instala PyTorch con soporte CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Luego instala el resto de dependencias:

```bash
pip install -r requirements.txt
```

---

## 📁 Estructura del Proyecto

```
📦 EmotionRecognitionModel
├── data_combined/         ← Dataset preprocesado (train/val)
├── models/                ← [model.py](models/model.py)
├── utils/                 ← [data_loader.py](utils/data_loader.py)
├── emotion_model.pth      ← Modelo entrenado
├── [train.py](train.py)               ← Entrenamiento
├── [evaluate.py](evaluate.py)         ← Evaluación
├── [main.py](main.py)                ← Inferencia
├── [viewer.py](viewer.py)            ← Visualización (opcional)
├── sample.jpg             ← Imagen de prueba
├── haarcascade_frontalface_default.xml ← Haar Cascade
├── [requirements.txt](requirements.txt)
└── README.md
```

---

## 🧠 Modelo

📄 [`models/model.py`](models/model.py)

CNN profunda con:
- 6 capas convolucionales
- BatchNorm y Dropout
- Global Avg Pooling + Dense layer
- 7 clases: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

---

## 📊 Entrenamiento

📄 [`train.py`](train.py)

```bash
python train.py
```

Guarda modelo en `emotion_model.pth` y muestra métricas por época.

---

## 🧪 Evaluación

📄 [`evaluate.py`](evaluate.py)

```bash
python evaluate.py
```

Evalúa el modelo cargando `emotion_model.pth`. Mide accuracy global y por clase.

---

## 🔍 Inferencia

📄 [`main.py`](main.py)

```bash
python main.py --image sample.jpg
```

1. Detecta rostro con Haar Cascade
2. Recorta y normaliza
3. Predice emoción y confianza

---

## 🖼️ Visualización

📄 [`viewer.py`](viewer.py)

Script opcional para mostrar resultados o métricas.

---

## 🧾 Datasets Utilizados

Este modelo fue entrenado utilizando una combinación de tres datasets públicos de reconocimiento de emociones faciales:

| Dataset | Descripción | Enlace |
|--------|-------------|--------|
| **CK+ (Extended Cohn-Kanade Dataset)** | Contiene imágenes etiquetadas con emociones básicas, ampliamente usado en estudios académicos. | [Ver en Kaggle](https://www.kaggle.com/datasets/shawon10/ckplus) |
| **FER-2013** | Dataset de emociones recolectado desde el desafío ICML 2013. Tiene más de 35,000 imágenes en escala de grises. | [Ver en Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| **Face Expression Recognition Dataset** | Otro conjunto adicional para mejorar la generalización y balancear clases. | [Ver en Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data) |

Los datos fueron unificados y reorganizados en `data_combined/` para facilitar el entrenamiento.
