import tkinter as tk
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Initialize the speech recognizer
r = sr.Recognizer()

def analyze():
    global r, model, tokenizer, output_label, text_box

    # Get the text from the text box
    text = text_box.get()

    # If the text box is empty, use the microphone
    if not text:
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source,timeout=5,phrase_time_limit=5)
            try:
                text = r.recognize_google(audio)
                print("You said: " + text)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")

                output_label.configure(text="Could not understand audio", font=("Arial", 24))
                return
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

                output_label.configure(text="Could not request results", font=("Arial", 24))
                return

    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Get the model output
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Get the probabilities and the predicted label
    probs = softmax(logits, dim=1)
    output_label_tensor = torch.argmax(probs)
    text=f"You said '{text}'\n Label:{model.config.id2label[output_label_tensor.item()]}, with score: {round(probs[0][output_label_tensor].item(), 4)}"
    text=text.title()

    # Update the output label
    output_label.configure(text=text, font=("Bahnschrift", 32))

# Create the root window
root = tk.Tk()
root.title("Output")
root.geometry("800x400")
root.configure(bg="#ADD8E6")

# Create the output label
output_label = tk.Label(root, text="Enter some text or click the button to start", font=("Bahnschrift", 24), bg="#ADD8E6")
output_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

# Create the text box
text_box = tk.Entry(root, font=("Bahnschrift", 24))
text_box.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Create the button
button = tk.Button(root, text="Analysis", font=("Bahnschrift", 24), command=analyze)
button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

# Start the main loop
root.mainloop()
