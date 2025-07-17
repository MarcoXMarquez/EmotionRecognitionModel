import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import torch
from torchvision import transforms
from models.model import EmotionCNN

model = EmotionCNN()
model.load_state_dict(torch.load("emotion_model.pth", weights_only=True))
model.eval()

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionApp:
    def __init__(self, root):
        self.root = root
        root.title("Reconocimiento de Emociones")

        self.emotion_label = tk.Label(root, text="", font=("Arial", 20))
        self.emotion_label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.upload_btn = tk.Button(btn_frame, text="Subir Imagen", command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=10)

        self.detect_btn = tk.Button(btn_frame, text="Detectar Emoción", command=self.detect_emotion)
        self.detect_btn.grid(row=0, column=1, padx=10)

        self.image_path = None
        self.display_image = None
        self.face_coords = None  # para guardar (x,y,w,h) de la cara detectada

    def upload_image(self):
        filetypes = (
            ("Archivos de imagen", "*.jpg *.jpeg *.png"),
            ("Todos los archivos", "*.*")
        )
        path = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=filetypes)
        if path:
            self.image_path = path
            self.show_image(path)
            self.emotion_label.config(text="")
            self.face_coords = None

    def show_image(self, path, draw_rect=False):
        img = Image.open(path).convert("RGB")

        # Si draw_rect es True y tenemos coords, dibujamos rectángulo
        if draw_rect and self.face_coords is not None:
            draw = ImageDraw.Draw(img)
            x, y, w, h = self.face_coords
            # Ajustamos grosor y color del rectángulo
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        img = img.resize((300, 300))
        self.display_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.display_image)

    def detect_emotion(self):
        if not self.image_path:
            messagebox.showerror("Error", "Primero debes subir una imagen.")
            return

        # Detectar cara y obtener coords
        img_cv = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            messagebox.showwarning("Advertencia", "No se detectó ninguna cara en la imagen.")
            return

        (x, y, w, h) = faces[0]
        self.face_coords = (x, y, w, h)

        face_img = img_cv[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])

        input_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)

        emotion = classes[pred.item()]
        self.emotion_label.config(text=f"Emoción detectada: {emotion}")

        # Mostrar imagen con recuadro
        self.show_image(self.image_path, draw_rect=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
