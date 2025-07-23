import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import cv2
import torch
from torchvision import transforms
from models.model import EmotionCNN
import os
from shutil import copyfile

# Cargar modelo
model = EmotionCNN()
model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
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

        self.folder_btn = tk.Button(btn_frame, text="Clasificar Carpeta", command=self.process_folder)
        self.folder_btn.grid(row=0, column=2, padx=10)

        self.camera_btn = tk.Button(btn_frame, text="Usar Cámara", command=self.capture_from_camera)
        self.camera_btn.grid(row=0, column=3, padx=10)

        self.progress = ttk.Progressbar(root, length=300, mode="determinate")
        self.progress.pack(pady=5)

        self.percent_label = tk.Label(root, text="0%")
        self.percent_label.pack()

        self.image_path = None
        self.display_image = None
        self.face_coords = None

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
        if draw_rect and self.face_coords is not None:
            draw = ImageDraw.Draw(img)
            x, y, w, h = self.face_coords
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        img = img.resize((300, 300))
        self.display_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.display_image)

    def detect_emotion(self):
        if not self.image_path:
            messagebox.showerror("Error", "Primero debes subir o capturar una imagen.")
            return

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

        self.show_image(self.image_path, draw_rect=True)

    def process_folder(self):
        folder_path = filedialog.askdirectory(title="Selecciona una carpeta de imágenes")
        if not folder_path:
            return

        output_base = os.path.join(folder_path, "clasificadas")
        os.makedirs(output_base, exist_ok=True)

        supported_ext = (".jpg", ".jpeg", ".png")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_ext)]

        total = len(files)
        if total == 0:
            messagebox.showwarning("Vacío", "No se encontraron imágenes válidas.")
            return

        self.progress["maximum"] = total
        self.progress["value"] = 0
        self.percent_label.config(text="0%")
        self.root.update()

        for idx, file in enumerate(files):
            full_path = os.path.join(folder_path, file)

            img_cv = cv2.imread(full_path)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                continue

            (x, y, w, h) = faces[0]
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
            emotion_dir = os.path.join(output_base, emotion)
            os.makedirs(emotion_dir, exist_ok=True)

            dest_path = os.path.join(emotion_dir, file)
            copyfile(full_path, dest_path)

            self.progress["value"] = idx + 1
            percent = int(((idx + 1) / total) * 100)
            self.percent_label.config(text=f"{percent}%")
            self.root.update()

        messagebox.showinfo("Procesamiento terminado", "Las imágenes han sido clasificadas por emoción.")

    def capture_from_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "No se pudo acceder a la cámara.")
            return

        messagebox.showinfo("Cámara", "Presiona 's' para capturar y 'q' para salir.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Presiona 's' para capturar", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                captured_path = "captured_image.jpg"
                cv2.imwrite(captured_path, frame)
                self.image_path = captured_path
                self.show_image(captured_path)
                self.emotion_label.config(text="")
                self.face_coords = None
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
