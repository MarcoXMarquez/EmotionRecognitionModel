import torch
from models.emotion_cnn import EmotionCNN

def show_model_architecture(model_path="emotion_model.pth", save_to_file=False):
    # Crear la instancia del modelo
    model = EmotionCNN()
    
    # Cargar los pesos entrenados
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Mostrar la arquitectura del modelo
    print("\n=== Arquitectura del Modelo ===")
    print(model)

    # Guardar la arquitectura a un archivo (opcional)
    if save_to_file:
        with open("model_architecture.txt", "w") as f:
            f.write(str(model))
        print("\nLa arquitectura fue guardada en 'model_architecture.txt'.")

if __name__ == "__main__":
    show_model_architecture()
