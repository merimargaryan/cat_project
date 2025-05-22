import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os

class CatDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat Detector")
        self.root.geometry("1000x800")
        
        # Проверяем наличие модели, если нет - скачиваем предобученную
        self.model_path = "cat_detection_yolov8n.pt"
        if not os.path.exists(self.model_path):
            messagebox.showinfo("Info", "Загрузка модели...")
            try:
                self.model = YOLO('yolov8n.pt')
                self.model.save(self.model_path)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не получилось загрузить модель: {str(e)}")
                return
        
        # Загрузка модели
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не получилось загрузить модель: {str(e)}")
            return
        
        # GUI
        self.create_widgets()
        
        # Переменные для хранения изображений
        self.original_image = None
        self.processed_image = None
        self.tk_image = None
    
    def create_widgets(self):
        # кнопки
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # загрузка изображения
        self.load_btn = tk.Button(
            button_frame, 
            text="Load Image", 
            command=self.load_image,
            width=15,
            height=2,
            font=('Arial', 12)
        )
        self.load_btn.pack(side=tk.LEFT, padx=10)
        
        # Кнопка детектива
        self.detect_btn = tk.Button(
            button_frame, 
            text="Detect Cat", 
            command=self.detect_cat,
            width=15,
            height=2,
            font=('Arial', 12)
        )
        self.detect_btn.pack(side=tk.LEFT, padx=10)
        self.detect_btn.config(state=tk.DISABLED)
        

        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=10)
        
        # Область для оригинального изображения
        self.original_label = tk.Label(image_frame, text="Original Image")
        self.original_label.pack(side=tk.LEFT, padx=10)
        
        # Область для обработанного изображения
        self.processed_label = tk.Label(image_frame, text="Processed Image")
        self.processed_label.pack(side=tk.LEFT, padx=10)
        
        # Результат
        self.result_label = tk.Label(
            self.root, 
            text="", 
            font=('Arial', 16),
            fg="black"
        )
        self.result_label.pack(pady=10)
        
        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                # Загрузка и отображение оригинального изображения
                self.original_image = Image.open(file_path)
                self.original_image.thumbnail((450, 450))
                self.tk_original = ImageTk.PhotoImage(self.original_image)
                
                self.original_label.config(image=self.tk_original)
                self.original_label.image = self.tk_original
                
                # Сброс обработанного изображения
                self.processed_label.config(image='')
                self.processed_label.image = None
                
                # Активация кнопки детекции
                self.detect_btn.config(state=tk.NORMAL)
                self.result_label.config(text="")
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def detect_cat(self):
        if not hasattr(self, 'original_image'):
            return
        
        try:
            self.status_var.set("Processing...")
            self.root.update()
            
            # Конвертация PIL Image в OpenCV format
            open_cv_image = cv2.cvtColor(
                cv2.imread(filedialog.askopenfilename()), 
                cv2.COLOR_RGB2BGR
            )
            
            # Детекция
            results = self.model.predict(open_cv_image)
            
            # Визуализация результатов
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Конвертация обратно в PIL Image
            self.processed_image = Image.fromarray(annotated_img)
            self.processed_image.thumbnail((450, 450))
            self.tk_processed = ImageTk.PhotoImage(self.processed_image)
            
            # Отображение обработанного изображения
            self.processed_label.config(image=self.tk_processed)
            self.processed_label.image = self.tk_processed
            
            # Определение есть ли кот
            cats_detected = False
            for result in results:
                for box in result.boxes:
                    if result.names[int(box.cls)] == 'cat':
                        cats_detected = True
                        break
            
            if cats_detected:
                self.result_label.config(text="Cat detected! 😺", fg="green")
            else:
                self.result_label.config(text="No cat detected. 😿", fg="red")
            
            self.status_var.set("Detection completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.status_var.set("Error during detection")

if __name__ == "__main__":
    root = tk.Tk()
    app = CatDetectorApp(root)
    
    # Центрирование окна
    window_width = 1000
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    root.mainloop()