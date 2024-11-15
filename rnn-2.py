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
import string
from argparse import ArgumentParser
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import seaborn as sns

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer,\
                          nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        _, hidden = self.rnn(inputs)
        # [to fill] obtain output layer representations
        output = self.W(hidden.view(-1, self.h))
        # [to fill] sum over output 
        #output_sum = torch.sum(output, dim=1)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)

        return predicted_vector


def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        testing = json.load(test_f)

    tra = []
    val = []
    tes = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in testing:
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

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0
    last_test_accuracy = 0

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

    test_log = {
        "epoch": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "conf_max": [],
    }

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        train_loss = 0

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                # Transform the input into required shape
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / (N // minibatch_size)
        train_acc = correct / total
        train_time = time.time() - start_time

        train_log['epoch'].append(epoch + 1)
        train_log['loss'].append(train_loss)                            # 记录平均训练损失
        train_log['accuracy'].append(train_acc)
        train_log['time'].append(train_time)

        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        trainning_accuracy = correct/total


        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        valid_loss = 0

        loss = None
        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))
            if loss is None:
                loss = example_loss
            else:
                loss += example_loss
            # print(predicted_label, gold_label)

        valid_loss = loss.item()/len(valid_data)
        valid_acc = correct / total
        valid_time = time.time() - start_time

        valid_log["epoch"].append(epoch + 1)
        valid_log["loss"].append(valid_loss)                    # 记录平均训练损失
        valid_log["accuracy"].append(valid_acc)
        valid_log["time"].append(valid_time)

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct/total

        model.eval()
        test_correct = 0
        test_total = 0
        random.shuffle(test_data)
        print("Test started for epoch {}".format(epoch + 1))
        test_data = test_data
        true_labels = []
        predicted_labels = []

        for input_words, gold_label in tqdm(test_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)  # output = predicted_vector
            true_labels.append(gold_label)
            predicted_labels.append(predicted_label)
            test_correct += int(predicted_label == gold_label)
            test_total += 1
            # print(predicted_label, gold_label)

        print("Test completed for epoch {}".format(epoch + 1))
        print("Test accuracy for epoch {}: {}".format(epoch + 1, test_correct / test_total))
        test_accuracy = test_correct/test_total

        conf_matrix = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
        test_log['conf_max'].append(conf_matrix)

        test_acc = test_accuracy
        test_precision = precision_score(y_true=true_labels, y_pred=predicted_labels, average="weighted")
        test_recall = recall_score(y_true=true_labels, y_pred=predicted_labels, average="weighted")
        test_f1 = f1_score(y_true=true_labels, y_pred=predicted_labels, average="weighted")

        test_log['epoch'].append(epoch + 1)                        # 记录平均训练损失
        test_log['accuracy'].append(test_acc)
        test_log['precision'].append(test_precision)
        test_log['recall'].append(test_recall)
        test_log['f1'].append(test_f1)

        if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy

        epoch += 1


    train_df = pd.DataFrame(train_log)
    valid_df = pd.DataFrame(valid_log)
    test_df = pd.DataFrame(test_log)

    train_df.to_csv(f"./{save_file_prefix}_train_log.csv", index=False)
    valid_df.to_csv(f"./{save_file_prefix}_valid_log.csv", index=False)
    test_df.to_csv(f"./{save_file_prefix}_test_log.csv", index=False)

    # 获取实际的 epoch 数
    actual_epochs = len(train_log["loss"])
    # 绘制损失曲线
    # Draw Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), train_log["loss"], label="Training Loss", marker=".")
    plt.plot(range(1, actual_epochs + 1), valid_log["loss"], label="Validation Loss", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f"./{save_file_prefix}_loss_curve.png")
    plt.show()

    # 绘制损失曲线
    # Draw Accuracy Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), train_log["accuracy"], label="Training Accuracy", marker=".")
    plt.plot(range(1, actual_epochs + 1), valid_log["accuracy"], label="Validation Accuracy", marker=".")
    plt.plot(range(1, actual_epochs + 1), test_log["accuracy"], label="Testing Accuracy", marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation and Testing Accuracy")
    plt.legend()
    plt.savefig(f"./{save_file_prefix}_accuracy_curve.png")
    plt.show()


    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
