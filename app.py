"""
program code
"""
from functools import lru_cache
import threading
from flask import Flask, render_template, request
from sentence_splitter import SentenceSplitter
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


app = Flask(__name__)

MODEL_NAME = 'tuner007/pegasus_paraphrase'
TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER = PegasusTokenizer.from_pretrained(MODEL_NAME)
MODEL = PegasusForConditionalGeneration.from_pretrained(
    MODEL_NAME).to(TORCH_DEVICE)

# Use LRU caching to store previously generated paraphrases
@lru_cache(maxsize=128)
def get_paraphrase(sentence, num_return_sequences):
    batch = TOKENIZER.prepare_seq2seq_batch(
        [sentence], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(TORCH_DEVICE)
    translated = MODEL.generate(**batch, max_length=60, num_beams=10,
                                num_return_sequences=num_return_sequences, temperature=2.0)
    tgt_text = TOKENIZER.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/paraphrase", methods=["POST"])
def paraphrase():
    context = request.form["context"]
    splitter = SentenceSplitter(language='en')
    sentence_list = splitter.split(context)

    paraphrases = []
    threads = []
    for sentence in sentence_list:
        # Generate 3 paraphrases per input text
        t = threading.Thread(target=paraphrase_thread, args=(sentence, 3, paraphrases))
        t.start()
        threads.append(t)

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Group paraphrases by index (1, 4, 7, 10, etc.)
    group1 = [paraphrases[i] for i in range(len(paraphrases)) if i % 3 == 0]
    group2 = [paraphrases[i] for i in range(len(paraphrases)) if i % 3 == 1]
    group3 = [paraphrases[i] for i in range(len(paraphrases)) if i % 3 == 2]

    return render_template("result.html", paraphrases=[group1, group2, group3])

def paraphrase_thread(sentence, num_return_sequences, paraphrases):
    # Check if paraphrases are already cached
    cached_paraphrases = get_paraphrase(sentence, num_return_sequences)
    paraphrases.extend(cached_paraphrases)

if __name__ == "__main__":
    app.run(debug=True)
