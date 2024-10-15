import os.path
import pickle


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    if not os.path.exists(name):
        return None
    with open(name , 'rb') as f:
        return pickle.load(f)


def save_text(name,text):
    with open(name, 'a') as f:
        f.write(text+'\n')