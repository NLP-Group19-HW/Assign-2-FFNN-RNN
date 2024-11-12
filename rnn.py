import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from argparse import ArgumentParser
import random
from tqdm import tqdm

# Constants
UNK = '<UNK>'

def load_glove_embeddings(filepath, embedding_dim=50):
    embeddings = {}
    with open(filepath, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=5):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = torch.sum(out, dim=0)  # Summing hidden states
        out = self.dropout(out)
        out = self.fc(out)
        return self.softmax(out)

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)

def load_data(train_path, val_path):
    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)
    train_data = [(review["text"].split(), int(review["stars"] - 1)) for review in train_data]
    val_data = [(review["text"].split(), int(review["stars"] - 1)) for review in val_data]
    return train_data, val_data

def hybrid_embedding(word, embeddings, embedding_dim):
    return embeddings.get(word.lower(), np.random.randn(embedding_dim).astype(np.float32))

def evaluate_model(model, data, embeddings, embedding_dim):
    model.eval()
    correct, total = 0, 0
    for words, label in data:
        vectors = np.array([hybrid_embedding(w, embeddings, embedding_dim) for w in words])
        vectors = torch.tensor(vectors).view(len(words), 1, -1)
        output = model(vectors)
        pred_label = output.argmax(dim=1).item()
        correct += (pred_label == label)
        total += 1
    return correct / total if total > 0 else 0

def train_rnn(args):
    embeddings = load_glove_embeddings(args.embeddings_path)
    train_data, val_data = load_data(args.train_data, args.val_data)

    model = RNN(input_dim=50, hidden_dim=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        random.shuffle(train_data)
        
        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i:i + args.batch_size]
            optimizer.zero_grad()
            batch_loss = 0
            
            for words, label in batch:
                vectors = np.array([hybrid_embedding(w, embeddings, 50) for w in words])
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                
                if output.dim() > 1:
                    pred_label = output.argmax(dim=1).item()
                else:
                    pred_label = output.argmax().item()
                
                loss = model.compute_loss(output.view(1, -1), torch.tensor([label]))
                batch_loss += loss
                total_loss += loss.item()
                correct += (pred_label == label)
                total += 1
            
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
        
        scheduler.step()

        train_accuracy = correct / total
        val_accuracy = evaluate_model(model, val_data, embeddings, 50)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data)}, Train Acc: {train_accuracy}, Val Acc: {val_accuracy}")

    print("Training complete. Final validation accuracy:", val_accuracy)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)  # Lower learning rate for fine-tuning
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    train_rnn(args)