import os
import speech_recognition as sr
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label, Canvas, Scrollbar, Frame


def audio_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak something...")
        audio = recognizer.listen(source)
        print("Recognizing...")
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return ""

def display_asl_images(text):
    print(f"Displaying ASL for: {text}")
    root = tk.Tk()
    root.title("ASL Representation")
    root.geometry("1080x480")

    asl_folder = "ASL_Images"
    image_refs = [] 
    max_width = 1080
    image_width = 100 
    padding = 10

    canvas = Canvas(root)
    canvas.pack(side="left", fill="both", expand=True)

    # Add a scrollbar to the canvas
    scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create a frame inside the canvas to hold the images
    frame = Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    # Function to update the scroll region
    def update_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", update_scroll_region)

    current_row = Frame(frame)  # Start with a new row
    current_row.pack(fill="x")
    current_width = 0  # Track the current width of the row

    for char in text.upper():
        if char.isalpha():
            img_path = os.path.join(asl_folder, f"{char}.png")
            print(f"Looking for image: {img_path}")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.resize((image_width, image_width))
                img_tk = ImageTk.PhotoImage(img)
                image_refs.append(img_tk)  # Prevent garbage collection

                # Check if adding the image exceeds the maximum width
                if current_width + image_width + padding > max_width:
                    # Start a new row
                    current_row = Frame(frame)
                    current_row.pack(fill="x")
                    current_width = 0

                # Add the image to the current row
                label = Label(current_row, image=img_tk)
                label.pack(side="left", padx=padding // 2)
                current_width += image_width + padding
            else:
                print(f"Image not found for: {char}")

    root.mainloop()

def main():
    text = audio_to_text()
    if text:
      
        with open("output.txt", "w") as f:
            f.write(text)
       
        display_asl_images(text)

if __name__ == "__main__":
    main()