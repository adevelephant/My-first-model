import torch
import torch.nn as nn
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

#Converting a question into numbers
def question_to_vector(question):
    vec = torch.zeros(vocab_size)
    for word in question.split():
        if word in word_to_id:
            vec[word_to_id[word]] = 1.0
    return vec

#Building the training tensors
X = torch.stack([question_to_vector(q) for q in questions])
y= torch.tensor([answer_to_id[a] for a in answers])

#The model
class SimpleQA(nn.Module):
    def __init__(self, voacb_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vocab_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

#Loss function and optimizer
model = SimpleQA(vocab_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

#Training loop
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    outputes = model(X)
    loss = criterion(outputes, y)
    loss.backward()
    optimizer.step()

    if(epoch + 1) % 50 == 0:
        print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")

#Asking the model the questions
def ask(question):
    model.eval()
    with torch.no_grad():
        vec = question_to_vector(question.lower()).unsqueeze(0)
        outputes = model(vec)
        predicted_id = outputes.argmax(dim=1).item()
        return id_to_answer[predicted_id]

if __name__ == "__main__":
    print("Ask me about a country's capital! (type 'quit' to stop)")
    while True:
        user_question = input("Your question: ")
        if user_question.lower() == "quit":
            break
        answer = ask(user_question)
        print("Answer:", answer)