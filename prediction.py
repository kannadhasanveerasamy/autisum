import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load your Keras models and define class names
model1 = keras.models.load_model('model/mobilevit_model.h5')
model2 = keras.models.load_model('model/model2.h5')
class_names1 = ['Autism Child', 'Non-Autism Child']
class_names2 = ['angry', 'happy', 'sad']

# Function to preprocess the image and make predictions using both models
def process_image(image_path):
    img = Image.open(image_path)
    
    # Preprocess the image for both models (resize to 224x224)
    img1 = img.resize((256, 256))
    img1 = np.array(img1)
    img1 = img1 / 255.0  # Normalize pixel values to [0, 1]

    # Make predictions using model1
    predictions1 = model1.predict(np.expand_dims(img1, axis=0))
    predicted_class1 = np.argmax(predictions1, axis=1)[0]

    img2 = img.resize((224, 224))
    img2 = np.array(img2)
    img2 = img2 / 255.0  # Normalize pixel values to [0, 1]


    # Make predictions using model2
    predictions2 = model2.predict(np.expand_dims(img2, axis=0))
    predicted_class2 = np.argmax(predictions2, axis=1)[0]

    # Update the class labels with the predicted class names from both models
    class_label2.config(text=f"Expression: {class_names2[predicted_class2]}")
    class_label1.config(text=f"RESULT: {class_names1[predicted_class1]}")
    

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((260,260))
        img = ImageTk.PhotoImage(img)
        image_label.img = img  # Save a reference to prevent garbage collection
        image_label.config(image=img)
        process_image(file_path)

app = tk.Tk()
app.title("AUTISM DISORDER CLASSIFICATION")
app.geometry("1200x800")  # Set window size

# Add a background image
background_image = Image.open('static\image4.jpg')  # Replace with the path to your background image
background_image = background_image.resize((1200, 800))
background_image = ImageTk.PhotoImage(background_image)
background_label = tk.Label(app, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Create and configure a label for displaying the selected image, centering it
image_label = tk.Label(app)
image_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

# Create a button for browsing images and center it
browse_button = ttk.Button(app, text="Browse Image", command=browse_image)
browse_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Create labels for displaying the results from both models and center them
class_label1 = ttk.Label(app, text="Result: ")
class_label1.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
class_label1.configure(font=("Arial", 30, "bold"), foreground="black")  # Set font size, style, and color

class_label2 = ttk.Label(app, text=" Expression: ")
class_label2.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
class_label2.configure(font=("Arial", 30, "bold"), foreground="black")  # Set font size, style, and color

# Add a heading title
#heading_label = tk.Label(app, text="AUTISM DISORDER CLASSIFICATION", font=("Arial", 16, "bold"))
#heading_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

app.mainloop()
