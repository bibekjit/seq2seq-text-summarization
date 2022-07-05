import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import contractions
import numpy as np

en_model = spacy.load('en_core_web_md')

def clean_text(text):
    toks = text.lower().split()
    toks = [contractions.fix(t) for t in toks]
    toks = en_model(" ".join(toks))
    toks = [str(t) for t in toks if (not t.is_punct) and (str(t)!="'s")]
    return " ".join(toks)

def non_rare_word_count(data,thresh=5):
    tk = Tokenizer()
    tk.fit_on_texts(data)
    count = np.array(list(dict(tk.word_counts).values()), dtype='int32')
    words = np.array(list(tk.word_counts), dtype='object')
    rare_words = len([i for i in range(len(words)) if count[i] < thresh])
    return len(tk.word_counts) - rare_words

def create_input_seq(text,xtk,text_maxlen):
    text = clean_text(text)
    seq = xtk.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=text_maxlen, padding='post')
    return seq.reshape((1, text_maxlen))





















