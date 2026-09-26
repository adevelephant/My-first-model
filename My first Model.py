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