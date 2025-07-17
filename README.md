
# Emotion Recognition CNN

Este proyecto implementa un modelo CNN desde cero en PyTorch para el reconocimiento de emociones faciales.

## Estructura del Proyecto
- `models/emotion_cnn.py`: definición del modelo
- `utils/data_loader.py`: carga y preprocesamiento
- `train.py`: entrenamiento del modelo
- `evaluate.py`: evaluación del modelo
- `main.py`: predicción con una imagen local

## Requisitos
Instala las dependencias con:

```
pip install -r requirements.txt
```

## Entrenamiento
```
python train.py
```

## Evaluación
```
python evaluate.py
```

## Predicción
Coloca una imagen en la raíz y cambia el nombre en `main.py`, luego ejecuta:

```
python main.py
```
