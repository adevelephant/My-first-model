import torch
import torch,nn as nn
import torch.optim as optim

#Training data
qa_pairs = [
    ("what is the capital of france", "Paris"),
    ("what is the capital of japan", "Tokyo"),
    ("what is the capital of italy", "Rome"),
    ("what is the capital of australia", "Canberra"),
    ("what is the capital of germany", "Berlin"),
    ("what is the capital of spain", "Madrid"),
    ("what is the capital of canada", "Ottawa"),
    ("what is the capital of egypt", "Cairo"),
    ("what is the capital of india", "New Delhi"),
    ("what is the capital of china", "Beijing"),
]

#Loops through each value
questions = [q for q, a in qa_pairs]
answers = [a for q, a in qa_pairs]

#Building vocab
all_words = set()
for q in questions:
    for word in q.split():
        all_words.add(word)
                      
vocab = sorted(all_words)
#Numbers each word
word_to_id = {word: i for i, word  in enumerate(vocab)}
vocab_size = len(vocab)

#Mapping answers to IDs
unique_answer = sorted(set(answers))
answer_to_id = {a: i for i, a in enumerate(unique_answer)}
id_to_answer = {i: a for a, i in answer_to_id.items()}
num_classes = len(unique_answer)