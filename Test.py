import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, filedialog, Canvas, PhotoImage
from PIL import Image, ImageTk

# Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Show only errors

# Load the trained model
model = load_model('our_model.h5')  # Ensure the path is correct
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Initialize the main window
root = Tk()
root.title("Pneumonia Detection")
root.geometry("600x500")

# Label to display the prediction result
result_label = Label(root, text="Please upload an image to analyze.", font=("Arial", 14))
result_label.pack(pady=20)

# Canvas to display the uploaded image
canvas = Canvas(root, width=224, height=224)
canvas.pack(pady=10)

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_data = preprocess_input(img_array)
    return img_data, img

# Function to make a prediction
def make_prediction():
    global img_path
    if img_path:
        img_data, img = preprocess_image(img_path)
        prediction = model.predict(img_data)
        
        # Update the result label based on the prediction
        if prediction[0][0] > prediction[0][1]:  # Assuming index 0 is 'NORMAL' and index 1 is 'PNEUMONIA'
            result_text = 'Person is safe.'
        else:
            result_text = 'Person is affected with Pneumonia.'
        
        result_label.config(text=result_text)

        # Display the image on the canvas
        img = img.resize((224, 224))
        tk_img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas.image = tk_img  # Keep a reference to avoid garbage collection

# Function to load an image file
def load_image():
    global img_path
    img_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
    if img_path:
        result_label.config(text="Image loaded. Click 'Analyze' to make a prediction.")

# Button to upload an image
upload_button = Button(root, text="Upload Image", command=load_image, font=("Arial", 12))
upload_button.pack(pady=10)

# Button to trigger the prediction
predict_button = Button(root, text="Analyze", command=make_prediction, font=("Arial", 12))
predict_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
