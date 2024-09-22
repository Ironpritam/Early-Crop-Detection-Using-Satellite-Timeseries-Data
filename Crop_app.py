import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import csv
import os
import subprocess

from PIL import Image, ImageTk
from model import *
from dataset_build import generate_csv_dataset  # Import the CSV dataset generation function



class ResultWindow(tk.Toplevel):
    def __init__(self, parent, accuracy, precision, recall, f1):
        super().__init__(parent)
        self.title("Test Results")
        self.geometry("300x200")
        self.configure(bg="#f0f0f0")

        label_result_accuracy = tk.Label(self, text=f"Accuracy: {accuracy:.2f}%", bg="#f0f0f0")
        label_result_accuracy.pack(pady=5)

        label_result_precision = tk.Label(self, text=f"Precision: {precision:.2f}", bg="#f0f0f0")
        label_result_precision.pack(pady=5)

        label_result_recall = tk.Label(self, text=f"Recall: {recall:.2f}", bg="#f0f0f0")
        label_result_recall.pack(pady=5)

        label_result_f1 = tk.Label(self, text=f"F1 Score: {f1:.2f}", bg="#f0f0f0")
        label_result_f1.pack(pady=5)

        self.button_save = tk.Button(self, text="Save Results", command=self.save_results,
                                      bg="#4CAF50", fg="white", relief=tk.GROOVE)
        self.button_save.pack(pady=10)

        # Set the window as transient to the parent window
        self.transient(parent)
        # Set the window as a priority window
        self.grab_set()
        # Wait for this window to be dealt with before going back to the parent window
        parent.wait_window(self)

    def save_results(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if filename:
            with open(filename, "w") as file:
                file.write(f"Accuracy: {accuracy:.2f}\n")
                file.write(f"Precision: {precision:.2f}\n")
                file.write(f"Recall: {recall:.2f}\n")
                file.write(f"F1 Score: {f1:.2f}\n")
            messagebox.showinfo("Save Results", "Results saved successfully!")
        else:
            messagebox.showinfo("Save Results", "No file selected!")

class CropTypeDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Crop Type Detector")
        self.master.geometry("1200x900")

        self.canvas = tk.Canvas(master, width=1200, height=900)
        self.canvas.pack(fill="both", expand=True)

        self.dataset_dir = ""
        self.model_path = ""
        self.model_trained = False
        self.model_loaded = False

        self.create_widgets()

        # Load the background image
        self.bg_image = Image.open("Images/background_image.jpg")
        self.bg_image = self.bg_image.resize((1200, 900), Image.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(self.bg_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.bg_image)

    def create_widgets(self):
        # Buttons on top row
        self.button_browse = tk.Button(self.canvas, text="Select Train/Test Dataset", command=self.browse_dataset,
                                       bg="#4CAF50", fg="white", relief=tk.GROOVE)
        self.button_browse.place(x=10, y=10)

        self.button_load_model = tk.Button(self.canvas, text="Load Model", command=self.load_model,
                                           bg="#4CAF50", fg="white", relief=tk.GROOVE)
        self.button_load_model.place(x=210, y=10)

        self.button_train = tk.Button(self.canvas, text="Train Model", command=self.train_model,
                                      bg="#008CBA", fg="white", relief=tk.GROOVE)
        self.button_train.place(x=360, y=10)

        self.button_test = tk.Button(self.canvas, text="Make Predictions", command=self.test_model,
                                     bg="#008CBA", fg="white", relief=tk.GROOVE)
        self.button_test.place(x=480, y=10)

        self.button_generate_csv = tk.Button(self.canvas, text="Generate CSV Dataset", command=self.generate_csv,
                                             bg="#ffcc00", fg="black", relief=tk.GROOVE)
        self.button_generate_csv.place(x=620, y=10)

        # self.label_accuracy = tk.Label(self.canvas, text="", bg="#f0f0f0")
        # self.label_accuracy.place(x=10, y=60)

        # self.label_precision = tk.Label(self.canvas, text="", bg="#f0f0f0")
        # self.label_precision.place(x=10, y=90)

        # self.label_recall = tk.Label(self.canvas, text="", bg="#f0f0f0")
        # self.label_recall.place(x=10, y=120)

        # self.label_f1 = tk.Label(self.canvas, text="", bg="#f0f0f0")
        # self.label_f1.place(x=10, y=150)

        self.button_reset = tk.Button(self.canvas, text="Reset", command=self.reset,
                                      bg="#f44336", fg="white", relief=tk.GROOVE)
        self.button_reset.place(x=790, y=10)
        
    def browse_dataset(self):
        self.dataset_dir = filedialog.askdirectory()
        if self.dataset_dir:
            messagebox.showinfo("Dataset Directory", "Dataset directory selected successfully!")
        else:
            messagebox.showerror("Error", "No dataset directory selected!")

    def load_model(self):
        # Implement loading prebuilt model here
        global model
        self.model_path = filedialog.askopenfilename()
        if self.model_path:          
            model = load_prebuilt_model(self.model_path)  # Load your prebuilt model
            messagebox.showinfo("Model Selection", "Model selected successfully!")
            print(model.summary())
            self.model_loaded = True
        else:
            messagebox.showerror("Error", "No model selected!")

    def train_model(self):
        if self.dataset_dir == "":
            messagebox.showerror("Error", "Please select a dataset directory!")
            return
        
        if self.model_loaded:
            response = messagebox.askquestion("Train Model", "Do you want to train the current loaded model further?")
            if response == 'yes':
                # Train the current loaded model further
                train_model(model, self.dataset_dir)
                self.model_trained = True
                messagebox.showinfo("Training", "Model trained successfully!")
            else:
                # Train a new model
                self.reset()
                messagebox.showinfo("New Training", "Please select dataset first and then click 'Train Model' button.")
        else:
            # Train a new model
            self.reset()
            messagebox.showinfo("New Training", "Please select dataset first and then click 'Train Model' button.")

    def test_model(self):
        if not (self.model_loaded or self.model_trained) :
            messagebox.showerror("Error", "Model not loaded or trained yet!")
            return

        # if :
        #     messagebox.showerror("Error", "Model not trained yet!")
        #     return

        # Call your function to test the model
        X, Y = load_data(self.dataset_dir)
        X_flat = X.reshape(X.shape[0], -1)

        scaler = MinMaxScaler()
        X_normalized_flat = scaler.fit_transform(X_flat)
        X = X_normalized_flat.reshape(X.shape)
        Y = to_categorical(Y)

        accuracy, precision, recall, f1 = test_model(model,X,Y)

        # Display results in a separate window
        ResultWindow(self.master, accuracy*100, precision, recall, f1)

    def reset(self):
        self.dataset_dir = ""
        self.model_trained = False
        self.model_loaded = False

        # self.label_accuracy.config(text="")
        # self.label_precision.config(text="")
        # self.label_recall.config(text="")
        # self.label_f1.config(text="")

        messagebox.showinfo("Reset", "All settings reset successfully!")
        
    def generate_csv(self):
        # Execute the script for generating CSV dataset
        try:
            generate_csv_dataset()  # Call the function to generate CSV dataset
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


def main():
    root = tk.Tk()
    app = CropTypeDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
