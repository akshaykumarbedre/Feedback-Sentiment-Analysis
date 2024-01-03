# Import the libraries and modules
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import speech_recognition as sr
import tkinter as tk

# Load a pre-trained model and tokenizer from Hugging Face
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a speech recognizer object
r = sr.Recognizer()

# Define a function that performs speech recognition and sentiment analysis
def analyze():
    # Use the global keyword to access the variables outside the function
    global r, model, tokenizer, output_label
    # Listen to the microphone and convert the voice to text
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source,timeout=5,phrase_time_limit=5)
        try:
            text = r.recognize_google(audio)
            print("You said: " + text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            # Update the label text with an error message
            output_label.configure(text="Could not understand audio", font=("Arial", 24))
            return
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            # Update the label text with an error message
            output_label.configure(text="Could not request results", font=("Arial", 24))
            return
    # Use the tokenizer to encode the text into input ids and attention masks
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # Use the model to make predictions on the encoded inputs and get the logits
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    # Use the softmax function to convert the logits into probabilities and get the label with the highest probability
    probs = softmax(logits, dim=1)
    output_label_tensor = torch.argmax(probs)
    text=f"You said '{text}'\n Label:{model.config.id2label[output_label_tensor.item()]}, with score: {round(probs[0][output_label_tensor].item(), 4)}"
    text=text.title()
    # Update the label text with the output of the sentiment analysis
    output_label.configure(text=text, font=("Bahnschrift", 32))

# Create a root window
root = tk.Tk()
root.title("Output")
root.geometry("800x400")
root.configure(bg="#ADD8E6")

# Create a label widget to display the output
output_label = tk.Label(root, text="Click the button to start", font=("Bahnschrift", 24), bg="#ADD8E6")
output_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

# Create a button widget to start the analysis
button = tk.Button(root, text="Analysis", font=("Bahnschrift", 24), command=analyze)
button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
output_label.configure(text="Click the button to start", font=("Bahnschrift", 32))

# Start the main loop
root.mainloop()
