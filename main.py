# Import the libraries and modules
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import speech_recognition as sr
import tkinter as tk



# Create a speech recognizer object
r = sr.Recognizer()

# Load a pre-trained model and tokenizer from Hugging Face
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Listen to the microphone and convert the voice to text
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Use the tokenizer to encode the text into input ids and attention masks
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Use the model to make predictions on the encoded inputs and get the logits
outputs = model(input_ids, attention_mask=attention_mask)
logits = outputs.logits

# Use the softmax function to convert the logits into probabilities and get the label with the highest probability
probs = softmax(logits, dim=1)
label = torch.argmax(probs)
text=f"label: {model.config.id2label[label.item()]}, with score: {round(probs[0][label].item(), 4)}"
text=text.capitalize()

# Create a root window
root = tk.Tk()

# Set the window title
root.title("Output")

# Set the window size
root.geometry("800x400")

# Set the window background color
root.configure(bg="#ADD8E6")
# Create a label widget to display the message
label = tk.Label(root, text=text, font=("Bahnschrift", 32))

# Place the label widget in the center of the window
label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Start the main loop
root.mainloop()