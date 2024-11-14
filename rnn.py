import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import random
import pickle

UNK = '<UNK>'

def load_glove_embeddings(filepath, embedding_dim=50):
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)
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

def load_data(train_path, val_path, test_path):
    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)
    train_data = [(review["text"].split(), int(review["stars"] - 1)) for review in train_data]
    val_data = [(review["text"].split(), int(review["stars"] - 1)) for review in val_data]
    test_data = [(review["text"].split(), int(review["stars"] - 1)) for review in test_data]
    return train_data, val_data, test_data

def hybrid_embedding(word, embeddings, embedding_dim):
    return embeddings.get(word.lower(), np.random.randn(embedding_dim).astype(np.float32))

def evaluate_model(model, data, embeddings, embedding_dim=50):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    start_time = time.time()
    with torch.no_grad():
        for words, label in data:
            vectors = np.array([hybrid_embedding(w, embeddings, embedding_dim) for w in words])
            vectors = torch.tensor(vectors).view(len(words), 1, -1)
            output = model(vectors)
            loss_sum += model.compute_loss(output.view(1, -1), torch.tensor([label])).item()
            pred_label = output.argmax(dim=1).item()
            correct += (pred_label == label)
            total += 1
    accuracy = correct / total if total > 0 else 0
    avg_loss = loss_sum / total if total > 0 else 0
    elapsed_time = time.time() - start_time
    return accuracy, avg_loss, elapsed_time

def evaluate_test_data(model, test_data, embeddings, embedding_dim=50):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for words, label in test_data:
            vectors = np.array([hybrid_embedding(w, embeddings, embedding_dim) for w in words])
            vectors = torch.tensor(vectors).view(len(words), 1, -1)
            output = model(vectors)
            pred_label = output.argmax(dim=1).item()
            y_true.append(label)
            y_pred.append(pred_label)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, conf_matrix

def train_and_evaluate(args):
    embeddings = load_glove_embeddings(args.embeddings_path)
    train_data, val_data, test_data = load_data(args.train_data, args.val_data, args.test_data)

    model = RNN(input_dim=50, hidden_dim=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = {
        "train": [],
        "validation": [],
        "test": None
    }

    print(f"{'Hidden Dim':<12} | {'Train Acc':<10} | {'Valid Acc':<10} | {'Train Loss':<10} | {'Valid Loss':<10} | {'Train Time':<10} | {'Valid Time':<10}")
    print("-" * 90)

    for epoch in range(args.epochs):
        # Training Phase
        model.train()
        start_time = time.time()
        train_acc, train_loss = run_training_epoch(model, train_data, optimizer, embeddings, args.batch_size)
        train_time = time.time() - start_time

        # Validation Phase
        val_acc, val_loss, val_time = evaluate_model(model, val_data, embeddings)
        
        # Print results for this epoch
        print(f"{args.hidden_dim:<12} | {train_acc:<10.4f} | {val_acc:<10.4f} | {train_loss:<10.4f} | {val_loss:<10.4f} | {train_time:<10.2f}s | {val_time:<10.2f}s")

        # Save results for each epoch
        results["train"].append({
            "epoch": epoch + 1,
            "accuracy": train_acc,
            "loss": train_loss,
            "time": train_time
        })
        results["validation"].append({
            "epoch": epoch + 1,
            "accuracy": val_acc,
            "loss": val_loss,
            "time": val_time
        })

    # Evaluate on test data
    test_acc, precision, recall, f1, conf_matrix = evaluate_test_data(model, test_data, embeddings)
    results["test"] = {
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist()
    }
    print(f"\nTest Results:\n")
    print(f"{'Metric':<10} | {'Score'}")
    print(f"{'-'*20}")
    print(f"Accuracy   | {test_acc:.4f}")
    print(f"Precision  | {precision:.4f}")
    print(f"Recall     | {recall:.4f}")
    print(f"F1 Score   | {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save results to JSON
    with open("rnn_results.json", "w") as f:
        json.dump(results, f, indent=4)

def run_training_epoch(model, train_data, optimizer, embeddings, batch_size):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    random.shuffle(train_data)

    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]
        optimizer.zero_grad()
        batch_loss = 0

        for words, label in batch:
            vectors = np.array([hybrid_embedding(w, embeddings, 50) for w in words])
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            pred_label = output.argmax(dim=1).item()
            loss = model.compute_loss(output.view(1, -1), torch.tensor([label]))
            batch_loss += loss
            total_loss += loss.item()
            correct += (pred_label == label)
            total += 1

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    return accuracy, avg_loss

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    train_and_evaluate(args)