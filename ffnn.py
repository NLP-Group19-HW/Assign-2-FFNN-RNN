import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import seaborn as sns


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden_layer = self.activation(self.W1(input_vector))        
        # [to fill] obtain output layer representation
        predicted_vector = self.W2(hidden_layer)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(predicted_vector)
        
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)  
    vocab_list.append(unk)      
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index       
        index2word[index] = word        
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])   
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)

    tra = []
    val = []
    tes = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in test:
        tes.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val, tes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    save_file_prefix = f"dim_{args.hidden_dim}"

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)     
    vocab, word2index, index2word = make_indices(vocab) 

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    test_data = convert_to_vector_representation(test_data, word2index)
    

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    print("========== Training for {} epochs ==========".format(args.epochs))

    train_log = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "time": [],
    }

    valid_log = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "time": [],
    }

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = 0
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data)
        with open("./train_results.out", "w") as f:
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    # 将 predicted_vector 和 predicted_label 写入文件
                    f.write(f"Predicted Vector: {predicted_vector.tolist()} \t Predicted Label: {predicted_label} \n")
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
                loss.backward()
                optimizer.step()
                train_loss += loss.item()   # calculate train_loss to draw

        train_loss = train_loss / (N // minibatch_size)
        train_acc = correct / total
        train_time = time.time() - start_time

        train_log['epoch'].append(epoch + 1)
        train_log['loss'].append(train_loss)    
        train_log['accuracy'].append(train_acc)
        train_log['time'].append(train_time)

        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_acc))
        print("Training time for this epoch: {}".format(train_time))

        valid_loss = 0
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16 
        N = len(valid_data)
        with open("./valid_results.out", "w") as f:
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    # 将 predicted_vector 和 predicted_label 写入文件
                    f.write(f"Predicted Vector: {predicted_vector.tolist()} \t Predicted Label: {predicted_label} \n")
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
                valid_loss += loss.item()

        valid_loss = valid_loss / (N // minibatch_size)
        valid_acc = correct / total
        valid_time = time.time() - start_time

        valid_log["epoch"].append(epoch + 1)
        valid_log["loss"].append(valid_loss)   
        valid_log["accuracy"].append(valid_acc)
        valid_log["time"].append(valid_time)

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, train_acc))
        print("Validation time for this epoch: {}".format(train_time))

        # valid_losses.append(valid_loss / (len(valid_data) // minibatch_size))  
        # print("Validation completed for epoch {}".format(epoch + 1))
        # print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        # print("Validation time for this epoch: {}".format(time.time() - start_time))

    train_df = pd.DataFrame(train_log)
    valid_df = pd.DataFrame(valid_log)

    train_df.to_csv(f"./{save_file_prefix}_train_log.csv", index=False)
    valid_df.to_csv(f"./{save_file_prefix}_valid_log.csv", index=False)

    # test
    correct = 0
    total = 0
    print("Test started")
    minibatch_size = 16
    N = len(test_data)
    true_labels = []
    predicted_labels = []

    with open(f"./{save_file_prefix}_test_results.out", "w") as f:
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            for example_index in range(minibatch_size):
                input_vector, gold_label = test_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                true_labels.append(gold_label)
                predicted_labels.append(predicted_label)
                # 将 predicted_vector 和 predicted_label 写入文件
                f.write(f"Predicted Vector: {predicted_vector.tolist()} \t Predicted Label: {predicted_label} \n")
                correct += int(predicted_label == gold_label)
                total += 1

    test_acc = correct / total
    conf_matrix = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)


    print("Test accuracy:", test_acc)
    print("Precision:", precision_score(y_true=true_labels, y_pred=predicted_labels, average="macro"))
    print("Recall:", recall_score(y_true=true_labels, y_pred=predicted_labels, average="macro"))
    print("F1-score:", f1_score(y_true=true_labels, y_pred=predicted_labels, average="macro"))
    print("Accuracy:", accuracy_score(y_true=true_labels, y_pred=predicted_labels))
    print("Confusion Matrix:", conf_matrix, sep="\n")

    # confusion matrix virualization
    plt.figure(figsize=(8, 6))
    labels = ["1", "2", "3", "4", "5"]
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"./{save_file_prefix}_confusion_matrix.png")
    plt.show()

    # draw loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_log["loss"], label="Training Loss", marker=".")
    plt.plot(range(1, args.epochs + 1), valid_log["loss"], label="Validation Loss", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f"./{save_file_prefix}_loss_curve.png")
    plt.show()

    # draw accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_log["accuracy"], label="Training Accuracy", marker=".")
    plt.plot(range(1, args.epochs + 1), valid_log["accuracy"], label="Validation Accuracy", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(f"./{save_file_prefix}_accuracy_curve.png")
    plt.show()

