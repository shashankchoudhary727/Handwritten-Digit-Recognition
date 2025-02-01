import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw
import os
from Prediction import predict, sigmoid

class DigitRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        
        # Set window size
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Handwritten Digit Recognition", 
            font=('Helvetica', 24, 'bold')
        )
        title_label.pack(pady=20)
        
        # Canvas frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(pady=20)
        
        # Drawing canvas (28x28 for MNIST format)
        self.canvas_size = 280
        self.canvas = tk.Canvas(
            canvas_frame, 
            width=self.canvas_size,
            height=self.canvas_size,
            bg='black',
            highlightthickness=2,
            highlightbackground='#333'
        )
        self.canvas.pack()
        
        # Create PIL image for drawing
        self.image = Image.new('L', (28, 28), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Draw a digit (0-9) in the box above",
            font=('Helvetica', 12)
        )
        instructions.pack(pady=10)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        # Clear button
        self.clear_btn = ttk.Button(
            button_frame,
            text="Clear Canvas",
            command=self.clear_canvas
        )
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Predict button
        self.predict_btn = ttk.Button(
            button_frame,
            text="Predict Digit",
            command=self.predict_digit
        )
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        # Prediction result
        self.result_label = ttk.Label(
            main_frame,
            text="Prediction will appear here",
            font=('Helvetica', 16)
        )
        self.result_label.pack(pady=20)
        
        # Drawing variables
        self.last_x = None
        self.last_y = None
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Load the trained model weights
        try:
            self.Theta1 = np.loadtxt('Theta1.txt')
            self.Theta2 = np.loadtxt('Theta2.txt')
        except FileNotFoundError:
            self.result_label.config(
                text="Error: Model weights not found!\nPlease train the model first.",
                foreground='red'
            )
            self.predict_btn.state(['disabled'])
    
    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_line(self, event):
        if self.last_x and self.last_y:
            # Draw on canvas
            x1, y1 = self.last_x, self.last_y
            x2, y2 = event.x, event.y
            self.canvas.create_line(x1, y1, x2, y2, width=20, fill='white', capstyle=tk.ROUND, smooth=True)
            
            # Scale coordinates to 28x28 image
            scale = 28 / self.canvas_size
            x1, y1 = int(x1 * scale), int(y1 * scale)
            x2, y2 = int(x2 * scale), int(y2 * scale)
            
            # Draw on PIL image
            self.draw.line([x1, y1, x2, y2], fill='white', width=2)
            
        self.last_x = event.x
        self.last_y = event.y
    
    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (28, 28), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction will appear here")
    
    def predict_digit(self):
        try:
            # Convert image to numpy array
            img_array = np.array(self.image)
            
            # Normalize and reshape
            img_array = img_array / 255.0
            img_array = img_array.reshape(1, 784)
            
            # Make prediction
            m = img_array.shape[0]
            a1 = np.append(np.ones((m, 1)), img_array, axis=1)
            z2 = np.dot(a1, self.Theta1.transpose())
            a2 = sigmoid(z2)
            a2 = np.append(np.ones((m, 1)), a2, axis=1)
            z3 = np.dot(a2, self.Theta2.transpose())
            a3 = sigmoid(z3)
            
            # Get prediction
            prediction = np.argmax(a3)
            
            # Update label
            self.result_label.config(
                text=f"Predicted Digit: {prediction}",
                foreground='blue'
            )
            
        except Exception as e:
            self.result_label.config(
                text=f"Error occurred: {str(e)}",
                foreground='red'
            )

def main():
    root = tk.Tk()
    app = DigitRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()