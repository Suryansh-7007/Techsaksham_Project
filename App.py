import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk  
import subprocess
import os
import csv

attendance_file = "attendance.csv"

def initialize_attendance_file():
    if not os.path.exists(attendance_file):
        with open(attendance_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Time"])

def capture_face_by_name():
    capture_window = tk.Toplevel(root)
    capture_window.title("Capture Face by Name")
    capture_window.geometry("300x200")

    tk.Label(capture_window, text="Enter Name:").pack(pady=5)
    name_entry = tk.Entry(capture_window)
    name_entry.pack(pady=5)

    def capture():
        name = name_entry.get().strip()
        if name:
            try:
                subprocess.run(["python", "data_capture.py", name], check=True)
                messagebox.showinfo("Success", f"Face Capture Complete for {name}")
                capture_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to capture face: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Name cannot be empty")

    tk.Button(capture_window, text="Capture", command=capture).pack(pady=10)

def train_model():
    try:
        subprocess.run(["python", "train_model.py"], check=True)
        messagebox.showinfo("Success", "Model Trained Successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Model training failed: {str(e)}")

def recognize_faces():
    import cv2
    from datetime import datetime

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("trained_model.yml")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    label_mapping = {0: "Unknown"}  
    with open("label_mapping.csv", mode="r") as label_file:
        for line in label_file:
            label, name = line.strip().split(',')
            label_mapping[int(label)] = name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to access camera")
        return

    recognized_names = set()
    messagebox.showinfo("Info", "Press 'q' to stop recognition")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            label, confidence = face_recognizer.predict(face)
            name = label_mapping.get(label, "Unknown")

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            
            if name != "Unknown" and confidence < 80:
                if name not in recognized_names:  
                    recognized_names.add(name)
                    now = datetime.now()
                    date = now.strftime("%Y-%m-%d")
                    time = now.strftime("%H:%M:%S")
                    with open(attendance_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name, date, time])
                    print(f"Marked attendance for: {name}")

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", f"Attendance marked for: {', '.join(recognized_names)}")

def manual_attendance():
    manual_window = tk.Toplevel(root)
    manual_window.title("Manual Attendance")
    manual_window.geometry("300x200")

    tk.Label(manual_window, text="Enter Name:").pack(pady=5)
    name_entry = tk.Entry(manual_window)
    name_entry.pack(pady=5)

    def save_attendance():
        name = name_entry.get().strip()
        if name:
            from datetime import datetime
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")
            with open(attendance_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, date, time])
            messagebox.showinfo("Success", "Attendance marked successfully")
            manual_window.destroy()
        else:
            messagebox.showwarning("Warning", "Name cannot be empty")

    tk.Button(manual_window, text="Submit", command=save_attendance).pack(pady=10)

def view_attendance():
    if os.path.exists(attendance_file):
        attendance_window = tk.Toplevel(root)
        attendance_window.title("Attendance Log")
        attendance_window.geometry("400x300")

        with open(attendance_file, mode='r') as file:
            content = file.read()

        text = tk.Text(attendance_window, wrap=tk.WORD)
        text.insert(tk.END, content)
        text.config(state=tk.DISABLED)
        text.pack(expand=True, fill=tk.BOTH)
    else:
        messagebox.showwarning("Warning", "No attendance log found")


initialize_attendance_file()

root = tk.Tk()
root.title("Attendance Management System")
root.geometry("800x600")  

bg_image = Image.open("bca.jpg")
bg_image = bg_image.resize((1600, 1400), Image.Resampling.LANCZOS)  
bg_photo = ImageTk.PhotoImage(bg_image)

canvas = tk.Canvas(root, width=800, height=600)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")
canvas.create_text(750, 50, text="Attendance Management System", font=("Helvetica", 55, "bold"), fill="orange")

def create_button(text, command, x, y, color):
    btn = tk.Button(root, text=text, command=command, bg=color, fg="green", font=("Helvetica", 30,"italic"))
    canvas.create_window(x, y, window=btn)

create_button("Capture Face by Name", capture_face_by_name, 750, 150, "green")
create_button("Train Model", train_model, 750, 250, "blue")
create_button("Recognize Faces", recognize_faces, 750, 350, "purple")
create_button("Manual Attendance", manual_attendance, 750, 400, "orange")
create_button("View Attendance Log", view_attendance, 750, 450, "brown")
create_button("Exit", root.destroy, 750, 600, "red")

root.mainloop()
