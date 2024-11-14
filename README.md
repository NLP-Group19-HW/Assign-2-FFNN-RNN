# starter code for a2

Add the corresponding (one) line under the ``[to fill]`` in ``def forward()`` of the class for ffnn.py and rnn.py

Feel free to modify other part of code, they are just for your reference.

---

One example on running the code:

**FFNN**

``python ffnn.py --hidden_dim 10 --epochs 1 ``
``--train_data ./training.json --val_data ./validation.json ``


**RNN**

``python rnn.py --hidden_dim 32 --epochs 10 ``
``--train_data training.json --val_data validation.json``

# 


Run experiments:

***Notice: Need to change file path for finding .json and .pkl

**FFNN**


``python ffnn.py --hidden_dim 5 --epochs 100 ``
``--train_data ./training.json --val_data ./validation.json --test_data ./test.json``

``python ffnn.py --hidden_dim 10 --epochs 100 ``
``--train_data ./training.json --val_data ./validation.json --test_data ./test.json``

``python ffnn.py --hidden_dim 20 --epochs 100 ``
``--train_data ./training.json --val_data ./validation.json --test_data ./test.json``


**RNN**

**Notice: Here do not have word_embedding.pkl file since it is too large  to be uploaded

``python rnn.py --hidden_dim 128 --epochs 20 ``
``--train_data training.json --val_data validation.json --embeddings_path ./word_embedding.pkl``
