import sys
import os
from tkinter import *
from PIL import ImageTk, Image  # Import from PIL to handle jpg images
import subprocess

window = Tk()
window.title("Driver Fatigue Detection in Vehicles using Computer Vision")

# Use PIL's Image and ImageTk to load .jpg image
background = "C:/Users/KISHORE/Downloads/hands-wheel-when-driving-high-speed-from-inside-car.jpg"
img = Image.open(background)  # Open the image file
photo = ImageTk.PhotoImage(img)  # Convert to PhotoImage for Tkinter

label_for_image = Label(window, image=photo)
label_for_image.place(x=0, y=0, relwidth=1, relheight=1)
label_for_image.pack()

# Set window size based on the image's dimensions
w, h = img.size
window.geometry(f'{w}x{h}')

# Function to run dcode.py using subprocess
def run():
    subprocess.run(['python', 'C:/Users/KISHORE/Downloads/Driver-Fatigue-Detection-in-Vehicles-using-Computer-Vision-main/Driver-Fatigue-Detection-in-Vehicles-using-Computer-Vision-main/src/dcode.py'])

# Button to initialize
b = Button(window, text="Initialize", command=run, height=4, width=10, justify=CENTER, font=(
    'calibri', 24, 'bold'), fg='black', bg='black')
b.place(relx=0.5, rely=0.5, anchor=CENTER)

window.mainloop()
