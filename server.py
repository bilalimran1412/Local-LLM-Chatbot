from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import time
import psutil
import os
from datetime import datetime

# Ensure protobuf is installed
try:
    import google.protobuf
except ImportError:
    raise ImportError(
        "The `protobuf` library is required but not installed. "
        "Please install it using `pip install protobuf`."
    )





# Load the text generation model
generation_model_name = 'google/flan-t5-large'  # Use a smaller model
generation_tokenizer = T5Tokenizer.from_pretrained(generation_model_name, legacy=False)  # Use new behavior
generation_model = T5ForConditionalGeneration.from_pretrained(generation_model_name)

# Function to get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

# Function to generate text
def generate_text(prompt):
    inputs = generation_tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = generation_model.generate(
            inputs['input_ids'],
            max_length=1024,  # Maximum length T5 typically supports
            min_length=100,   # Ensure responses aren't too short
            num_beams=4,      # Beam search for better quality
            length_penalty=2.0,  # Encourage longer responses
            no_repeat_ngram_size=3  # Avoid repetition
        )
    return generation_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize Flask app
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    incoming_time = datetime.now()
    print(f"Incoming request at: {incoming_time.isoformat()}")  # Log incoming time
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    data = request.json
    prompt = data.get('prompt', '')
    print(f"Prompt received: {prompt}")  # Log received prompt
    response = generate_text(prompt)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    outgoing_time = datetime.now()
    print(f"Outgoing response at: {outgoing_time.isoformat()}")  # Log outgoing time
    print(f"Generation time: {end_time - start_time:.2f}s | Memory usage: {end_memory - start_memory:.2f}MB")
    print(f"Generated response: {response}")  # Log generated response
    return jsonify({'response': response})

if __name__ == '__main__':
    print("Starting Flask server...")  # Debugging statement
    app.run(host='0.0.0.0', port=5000)
    print("Flask server started")  # Debugging statement



