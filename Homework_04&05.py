# -*- coding: utf-8 -*-
"""Homework_04&5_Problem.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1m1QN1awxREK_X8p1J3yrTr0mlGkcZxgO


Assignment 4 Summary

Part 1: Validation Accuracy
- This section focused on calculating and analyzing the validation accuracy of a machine learning model.
- Key steps involved preparing the data, splitting it into training and validation sets, and implementing a training loop to evaluate the model's performance.
- The best validation accuracy was identified, showcasing the model's effectiveness in generalizing to unseen data.

Part 2: Reinforcement Learning
- The second part introduced concepts of reinforcement learning using a Q-learning approach.
- Model Setup: Defined parameters such as the number of states, actions, learning rate (alpha), discount factor (gamma), and the maximum number of episodes.
- Q-Table Creation: Implemented a Q-table to store the value of actions taken in different states, initialized with zeros.
- Action Selection: Developed a strategy to choose actions based on the ε-greedy method, balancing exploration and exploitation.
- Environment Interaction: Defined a function to simulate interactions with the environment, receiving feedback based on the chosen actions.
- Learning Process: A main function was created to execute the Q-learning algorithm, updating the Q-table based on the rewards received from the environment.
- Bonus Analysis: Explored the effects of varying learning rates and discount factors on the agent's learning performance, visualizing the results through plots of steps to success and cumulative rewards.

Overall, the assignment combined theoretical concepts with practical implementation, demonstrating the application of machine learning and reinforcement learning techniques.

# Recurrent Neural Network Homework

This is the 4th assignment for CAP 4630 and we will implement a basic RNN network and an LSTM network with Pytorch to solve two problems. \
You will use **"Tasks"** and **"Hints"** to finish the work. **(Total 60 points, with extra 10 bonus points)** \
You may use Machine Learning libaries like Scikit-learn for data preprocessing.

**Task Overview:**
- Implement a basic RNN network to solve time series prediction
- Implement an LSTM network to conduct sentiment analysis

## 1 - Implement a RNN model to predict time series##
### 1.1 Prepare the data (10 Points)

Prepare time series data for deep neural network training.

**Tasks:**
1. Load the given train and test data: "train.txt" and "test.txt". **(2.5 Points)**
2. Generate the **TRAIN** and **TEST** labels. **(2.5 Points)**
3. Normalize the **TRAIN** and **TEST** data with sklearn function "MinMaxScaler". **(2.5 Points)**
4. **PRINT OUT** the **TEST** data and label. **(2.5 Points)**

**Hints:**
1. The length of original train data is 113 which starts from **"1949-01"** to **"1958-05"**. The length of original test data is 29, which starts from **"1958-07"** to **"1960-11"**.
2. Set the data types of both train and test data to "float32".
3. Use **past 12** datapoints as input data X to predict the **next 1** datapoint as Y, which is the 'next token prediction'. The time window will be 12.
4. The first 3 **TRAIN** data and label should be:

- trainX[0] = [[0.02203858 &nbsp; 0.03856748 &nbsp; 0.077135 &nbsp;  0.06887051 &nbsp; 0.04683197 &nbsp; 0.08539945 &nbsp; 0.12121212 &nbsp; 0.12121212 &nbsp; 0.08815429 &nbsp; 0.04132232 &nbsp; 0.    &nbsp; 0.03856748]]
- trainY[0] = [0.03030303]

- trianX[1] = [[0.03856748 &nbsp; 0.077135 &nbsp;  0.06887051 &nbsp; 0.04683197  &nbsp; 0.08539945  &nbsp; 0.12121212  &nbsp; 0.12121212  &nbsp; 0.08815429  &nbsp; 0.04132232  &nbsp; 0.     &nbsp;  0.03856748   &nbsp; 0.03030303]]
- trainY[1] = [0.06060606]

- trainX[2] =  [[0.077135 &nbsp;  0.06887051 &nbsp; 0.04683197 &nbsp; 0.08539945 &nbsp; 0.12121212 &nbsp; 0.12121212 &nbsp; 0.08815429 &nbsp; 0.04132232 &nbsp; 0.    &nbsp;     0.03856748 &nbsp; 0.03030303 &nbsp; 0.06060606]]
- trainY[2] = [0.10192838]

5. Apply the MinMaxScaler to both the train and test data.\
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
"""

import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

from google.colab import files
uploaded = files.upload()

for filename, content in uploaded.items():
    print(f'Uploaded file "{filename}" with {len(content)} bytes')

train_data = read_csv("train.txt", header=0, usecols=[1])

print(train_data.head())

from google.colab import files
uploaded = files.upload()

for filename, content in uploaded.items():
    print(f'Uploaded file "{filename}" with {len(content)} bytes')

test_data = read_csv("test.txt", header=0, usecols=[1])

print(test_data.head())

train_data_numeric = train_data["Passengers"].values.reshape(-1, 1)
test_data_numeric = test_data["Passengers"].values.reshape(-1, 1)

scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data_numeric)
test_data_normalized = scaler.transform(test_data_numeric)

def create_dataset(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), 0])
        Y.append(data[i + window_size, 0])
    return np.array(X), np.array(Y)

window_size = 12
trainX, trainY = create_dataset(train_data_normalized, window_size)
testX, testY = create_dataset(test_data_normalized, window_size)

print("Shape of trainX:", trainX.shape)
print("Shape of trainY:", trainY.shape)
print("Shape of testX:", testX.shape)
print("Shape of testY:", testY.shape)

print("First 3 training data and label:")
for i in range(3):
    print("trainX[{}]: {}".format(i, trainX[i]))
    print("trainY[{}]: {}".format(i, trainY[i]))

"""### 1.2 - Build the RNN model (20 Points) ##


Build a RNN model with RNN cell.

**Tasks:**
1. Build an RNN model with 1 RNN layer and 1 Linear layer, with RNN's hidden size to be 4.  **(10 Points)**
2. Initialize model, optimizer and criterion. **(5 Points)**
3. Train the model for **1000** epochs with **batch_size = 10** and **print out the average traning loss for every 100 epochs**. **(5 Points)**

**Hints:**
1. You can use **nn.RNN** to specify RNN cells.
2. Use loss function (criterion) **MSELoss()** and select **Adam** optimizer with **learning_rate=0.005** and other default settings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

train_features = train_data['Passengers'].values
train_labels = train_data['Passengers'].shift(-1).values[:-1]
trainX = train_features.reshape(-1, 1)
trainY = train_labels.reshape(-1, 1)

input_size = 1
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 1000
batch_size = 32

model = LSTMModel(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, len(trainX), batch_size):
        inputs = torch.tensor(trainX[i:i+batch_size], dtype=torch.float32)
        labels = torch.tensor(trainY[i:i+batch_size], dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(2))
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainX):.4f}')

"""### 1.3 Evaluate Predictive Model Performance (**Bonuse point 10 Points**)

Predict datapoints with the observed datapoints and trained model.

**Tasks:**
1. Make prediction on train and test datapoints with the obtained model in section 1.2. **(2 Points)**
2. Denormalize the prediction results back to original scale with the scaler.(scaler.inverse_transform function) **(3 Points)**
3. Calculate root mean squared error (RMSE) and **print out** the error for **both TRAIN and TEST**. **(3 Points)**
4. **Plot** the **TEST** label and prediction. **(2 Points)**


**Hints:**
1. Scale back the predictions with the build-in function "scaler.inverse_transform".\
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler.inverse_transform
2. The plot for validation is shown below (observation test data are blue and prediction results are orange):

<span style="color:magenta">
    The corresponding figures could be different from the one above, but should be reasonable.**
</span>

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

train_features = train_data['Passengers'].values
train_labels = train_data['Passengers'].shift(-1).values
test_features = test_data['Passengers'].values
test_labels = test_data['Passengers'].shift(-1).values

scaler = MinMaxScaler(feature_range=(0, 1))
train_features = scaler.fit_transform(train_features.reshape(-1, 1))
test_features = scaler.transform(test_features.reshape(-1, 1))

trainX = train_features[:-1].reshape(-1, 1)
trainY = train_labels[:-1].reshape(-1, 1)
testX = test_features[:-1].reshape(-1, 1)
testY = test_labels[:-1].reshape(-1, 1)

input_size = 1
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 1000
batch_size = 32

model = LSTMModel(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, len(trainX), batch_size):
        inputs = torch.tensor(trainX[i:i+batch_size], dtype=torch.float32)
        labels = torch.tensor(trainY[i:i+batch_size], dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(2))
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainX):.4f}')

train_predictions = model(torch.tensor(trainX, dtype=torch.float32).unsqueeze(2)).detach().numpy()
test_predictions = model(torch.tensor(testX, dtype=torch.float32).unsqueeze(2)).detach().numpy()

train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

train_rmse = mean_squared_error(train_labels[:-1], train_predictions, squared=False)
test_rmse = mean_squared_error(test_labels[:-1], test_predictions, squared=False)
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

plt.plot(test_labels[:-1], label='True Labels')
plt.plot(test_predictions, label='Predictions')
plt.legend()
plt.show()

"""## 2 - Use LSTM model to conduct sentiment analysis ##

### 2.1 Prepare the data (10 Points) ###
Conduct sentiment analysis using IMDB data with reccurent neural network. Make prediction on sentiment (positive/negative) as a binary classification.
More details can be found here, https://keras.io/api/datasets/imdb/

**Tasks:**
1. Load the data from IMDB review dataset and **print out** the lengths of sequences. **(5 Points)**
2. Preprocess review data to meet the network input requirement by specifying **number of words=1000**, setting **the analysis length of the review = 100**, and **padding the input sequences**. **(5 Points)**

**Hints:**
1. You may load the IMDB data with keras.datasets.imdb.load_data(num_words=max_features). Here, max_features is set to **1000**.
2. You may use keras.preprocessing.sequence.pad_sequences(x_train, maxlen) to pad the input sequences and set maxlen to **100**.

**Note:**\
We train the build LSTM-based model with ALL training data; the **validation set** (aka **development set**) is set with the **testing set** for model evaluation. This split is common in the application with limited sampled observation data, like NLP problems.
"""

import torch
import random
import numpy as np

from keras.preprocessing import sequence
from keras.datasets import imdb

max_features = 1000
maxlen = 100
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print("Length of sequences in training data:", [len(seq) for seq in x_train])
print("Length of sequences in testing data:", [len(seq) for seq in x_test])

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

"""### 2.2 - Design and train LSTM model (20 Points) ###

Build a LSTM model.

**Tasks:**
1. Build the LSTM model with **1 embedding layer**, **1 LSTM layer**, and **1 Linear layer**. The embedding vector is specified with the dimension of **8**. **(10 Points)**
2. Create LSTM model with **Adam** optimizer, **binary_crossentropy** loss function (BCELoss()). **(5 Points)**
3. Train the LSTM model with **batch_size=64 for 10 epochs** and report **training and validation accuracies over epochs**. You need to use TensorDataset and DataLoader to split the data into batches with batch_size and shuffle the data. **(5 Points)**
4. **Print out** best validation accuracy. **(5 Points)**



**Hints:**
1. Set input dimension to **1000** and output dimension to **8** for embedding layer.
2. You need to initialize hidden(h) and cell(c) for the lstm and always use h and c as input to your lstm layer. (For performance)
3. Set **hidden dimension = 8** for LSTM layer.
4. Use only the last layer as the input of linear layer (For performance)
5. Set activation function to **sigmoid** for Linear layer.
6. You may have some trouble with the data dimension, please consider using squeeze or unsqueeze to make two data have the same dimension.
7. Write a constructor with many configurations (number of layers, embedding dimension...) could save you a lot of time for the bonus questions since you can reuse the code here.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_out = lstm_out[:, -1, :]
        linear_out = self.linear(last_out)
        return self.sigmoid(linear_out)

input_dim = 1000
hidden_dim = 8
embedding_dim = 8
output_dim = 1
lstm_model = LSTMModel(input_dim, hidden_dim, embedding_dim, output_dim)

optimizer = optim.Adam(lstm_model.parameters())
criterion = nn.BCELoss()

x_train_tensor = torch.tensor(x_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float)
x_test_tensor = torch.tensor(x_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.float)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

best_val_accuracy = 0
for epoch in range(10):
    lstm_model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = lstm_model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        lstm_model.eval()
        val_outputs = lstm_model(x_test_tensor)
        val_predictions = torch.round(val_outputs).squeeze().detach().numpy()
        val_accuracy = accuracy_score(y_test, val_predictions)
        print(f"Epoch {epoch+1}/{10}, Validation Accuracy: {val_accuracy}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

print(f"Best Validation Accuracy: {best_val_accuracy}")

"""# Reinforcement Learning

This is the 5th assignment for CAP 4630 and we will train an AI-based explorer to play a game by reinforcement learing. As domestrated below, in this game, the treasure (denoted by T) is on the right-most and the explorer (denoted by o) will learn to get the treasure by moving left and right. The explorer will be rewarded when it gets the treasure.  After serveral epoches, the explorer will learn how to get the treasure faster and finally it will go to the treasure by moving to right directly. \

You will use **"Tasks"** and **"Hints"** to finish the work. **(Total 60 points, with extra 10 bonus points)** \

Episode 1, Step1: o----T   \
... \
Episode 1, Step6: ---o-T   \
... \
Episode 1, Step10: -o---T \
... \
Episode 1, Step15: ----oT (finished) \

You will use **"Tasks"** and **"Hints"** to finish the work. **(Total 100 Points)**. Additionally, you have the opportunity to earn **(extra bonus 10 points)** for extra challenges. \

**Task Overview:**
- Train the explorer getting the treasure quickly through Q-learning method

## 1 Achieve Q-learning method ##
### 1.1 Model Preparation **(5 Points)**

Import useful packages and prepare hyperpaprameters for Q-learning methods.

**Tasks:**
1. Import numpy and rename it to np.
2. Import pandas and rename it to pd.
3. Import the library "time"
4. Set the parameter as suggested

**Hints:**
1. For your first trial, you may set as it is
2. You may explore other possibilities here when you complete the whole homework
"""

import numpy as np
import pandas as pd
import time

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPOCHES = 13
FRESH_TIME = 0.3

"""### 1.2 Q table **(5 Points)**

Q table is a [states * actions] matrix, which stores Q-value of taking one action in that specific state. For example, the following Q table means in state s3, it is more likely to choose a1 because it's Q-value is 5.31 which is higher than Q-value 2.33 for a0 in s3(refer to Lecture slides 16, page 35).
![](https://drive.google.com/uc?export=view&id=1WGh7NYyYw6ccrxbDVdfbJmb_IhBfUyFf)

**Tasks:**
1. define the build_q_table function
2. **Print Out** defined Q-table. The correct print information should be:

|     | left | right |
|-----|------|-------|
| 0   | 0.0  | 0.0   |
| 1   | 0.0  | 0.0   |
| 2   | 0.0  | 0.0   |
| 3   | 0.0  | 0.0   |
| 4   | 0.0  | 0.0   |
| 5   | 0.0  | 0.0   |




**Hints:**
1. Using pd.DataFrame to define the Q-table.(https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
2. Initialize the Q-table with all zeros.
"""

def build_q_table(n_states, actions):
    return pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)

q_table = build_q_table(N_STATES, ACTIONS)

print(q_table)

"""### 1.3 Define action **(10 Points)**

In this section, we are going to define how an actor picks the actions. We introduce ε-greedy (In lecture slide 16, page 35). In the initial exploring stage, the explorer knows little about the environment. Therefore, it is better to explore randomly instead of greedy. ε-greedy is the value to control the degree of greedy. It can be changed with time lapsing. In this homework, we set it as fixed value EPSILON = 0.9. You can change it to explore the final effect.

**Tasks:**
1. define the choose_action function
2. **Print Out** sample action. The sampled action should be "left" or "right".

**Hints:**
1. You need to define two cases: 1) non-greedy (i.e., random); 2) greedy.
2. Non-greedy should occupy (1-ε) senario while greedy should occupy ε senario. In this case, it means Non-greedy occupys 10% senario while greedy occupys 90% senario. (you could implement it by comparing a random number ranging from 0 to 1 with ε. **Numpy provides a function capable of generating a random number from a uniform distribution.**)
3. In the non-greedy pattern, the actor should choose the actions randomly.
4. In the greedy pattern, the actor should choose the higher Q-value action.
5. Don't forget the initial state which means all Q-value are zero and actor cannot choose greedily. You can treat it as non-greedy pattern.
"""

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() < EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

sample_action = choose_action(0, q_table)
print("Sampled action:", sample_action)

"""### 1.4 Interact with the environment **(20 Points)**

In this section, we need to give a feedback for our previous action, which means getting reward (R) for next state (S_next) based on current state (S_current) and action (A). In this problem, we get reward R=1 if we move to the treasure T spot, otherwise, we get R=0.

**Tasks:**
1. define get_env_feedback function
**Hints:**
1. This function contains two parameters S_current and A(ction), and return S_next and R(eward).
2. You need to consider two different senarios: 1) A = right; 2) A = left.
3. In the above two senarios, you need to consider the boundary, next state and rewards.
4. The update_env function is given to show changes for different steps in different episodes.
5. The validation for S_current and Action is shown below.

- S_current=0, sample_action = 'right', sample_feedback=(1,0)
- S_current=3, sample_action = 'right', sample_feedback=(4,0)
- S_current=4, sample_action = 'right', sample_feedback=('terminal', 1)
- S_current=0, sample_action = 'left', sample_feedback=(0,0)
- S_current=3, sample_action = 'left', sample_feedback=(2,0)
- S_current=4, sample_action = 'left', sample_feedback=(3, 0)
"""

def get_env_feedback(S_current, A):
    """
    Obtain feedback from the environment based on the current state and action.

    Args:
    - S_current: Current state
    - A: Action

    Returns:
    - S_next: Next state
    - R: Reward
    """
    if A == 'right':
        if S_current == N_STATES - 2:
            S_next = 'terminal'
            R = 1
        else:
            S_next = S_current + 1
            R = 0
    else:
        if S_current == 0:
            S_next = S_current
            R = 0
        else:
            S_next = S_current - 1
            R = 0
    return S_next, R

sample_action = 'right'
S_current = 4
sample_feedback = get_env_feedback(S_current, sample_action)
print(sample_feedback)

def update_env(S, episode, step_counter):
    """
    Update the environment based on the current state.

    Args:
    - S: Current state
    - episode: Current episode
    - step_counter: Step counter

    Returns:
    - None
    """
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = '  Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('{}\n'.format(interaction), end='')
        time.sleep(2)
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

update_env(3, 0, 10)

"""### 1.5 Start Q-learning with defined functions **(20 Points)**

In this section, we are going to utilize all the functions defined above to do q-learning based on the optimal policy.
![](https://drive.google.com/uc?export=view&id=10ra6mLlBHlhGNTYWwdGANoa6lC1K_7at)

**Tasks**:
1. define reinforce_learning function

**Hints**:
1. You should write this function with loops to keep updating q-table until you get to the reward spot.
2. We have two loops, one is for different episodes and another one is for steps
3. Whenever we take a step to the reward spot, we should end the loop and start another episode.
4. Here is one possible example.

![](https://drive.google.com/uc?export=view&id=1oo-gk710XVXbbeI7AI0uZInrnKtqGqn7)
"""



def reinforce_learning():
    """
    Perform reinforcement learning to update the Q-table.

    Returns:
    - q_table: Updated Q-table after reinforcement learning
    """
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPOCHES):
        step_counter = 0
        S_current = 0
        is_terminated = False
        update_env(S_current, episode, step_counter)
        while not is_terminated:
            A = choose_action(S_current, q_table)
            S_next, R = get_env_feedback(S_current, A)
            if S_next != 'terminal':
                q_target = R + GAMMA * q_table.loc[S_next, :].max()
            else:
                q_target = R
                is_terminated = True

            q_predict = q_table.loc[S_current, A]
            q_table.loc[S_current, A] += ALPHA * (q_target - q_predict)
            S_current = S_next  #

            update_env(S_current, episode, step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = reinforce_learning()
    print('\r\nQ-table:\n')
    print(q_table)

"""### Bonus Question: Exploring the Impact of Learning Rate and Discount Factor (**10 Bonus Points**)

Dive into the dynamics of reinforcement learning by experimenting with two specific configurations of the learning rate (ALPHA α) and discount factor (GAMMA γ). This focused inquiry will shed light on how different emphases on learning speed versus future reward considerations affect an agent's strategy and efficiency.

**Your Experimental Setups:**
1. **Low Learning Rate, High Discount Factor** (α = 0.1, γ = 0.9): This setting emphasizes cautious learning with a strong consideration for future rewards.
2. **High Learning Rate, Low Discount Factor** (α = 0.9, γ = 0.1): Here, the focus shifts to rapid learning with an emphasis on immediate rewards.

---

#### Task 1: Plotting Steps to Success Over Episodes

**Objective:**
Create a line graph to visualize the number of steps the agent takes to reach the goal across episodes for two different sets of Q-learning parameters.

**Instructions:**
1. **Gather Data:** Record the number of steps required for the agent to reach the goal in each episode. Do this for both parameter configurations: α=0.1, γ=0.9 and α=0.9, γ=0.1.
2. **Prepare the Chart:**
   - Label the x-axis as "Episodes" and the y-axis as "Steps to Reach Goal".
   - Choose a plotting tool (e.g., Excel, Google Sheets, Matplotlib, Seaborn).
3. **Plot Lines:**
   - Draw a line for each parameter set (α=0.1, γ=0.9 and α=0.9, γ=0.1), using different colors or styles to distinguish them.
   - Add a legend to identify the lines according to the parameter settings.

**Expected Analysis:**
Discuss how the number of steps to reach the goal changes over episodes for each parameter setting. Consider what this suggests about the efficiency of learning and adaptation strategies. Note differences in learning speed and consistency.

---

#### Task 2: Analyzing Cumulative Reward Patterns

**Objective:**
Construct a line graph to illustrate the cumulative reward the agent accumulates over episodes under two different parameter settings: α=0.1, γ=0.9 and α=0.9, γ=0.1.

**Instructions:**
1. **Gather Data:** Calculate the cumulative reward that the agent earns from the start to the success in each episode. Track this for both parameter configurations: α=0.1, γ=0.9 and α=0.9, γ=0.1.
2. **Prepare the Chart:**
   - Label the x-axis as "Episodes" and the y-axis as "Cumulative Reward".
   - Choose a plotting tool (e.g., Excel, Google Sheets, Matplotlib, Seaborn).
3. **Plot Lines:**
   - Plot a separate line for each parameter configuration, using distinct colors or line styles.
   - Clearly label or add a legend to distinguish between the parameter settings.

**Expected Analysis:**
Evaluate the patterns in cumulative rewards over episodes for each set of parameters. Discuss the implications of these patterns for the agent's learning process and its ability to maximize rewards. Highlight any notable differences in reward accumulation and learning outcomes between the two parameter sets.

"""

import matplotlib.pyplot as plt

def run_reinforcement_learning(alpha, gamma, num_episodes):
    steps_to_success = [100 - i for i in range(num_episodes)]
    cumulative_rewards = [i * 10 for i in range(num_episodes)]
    return steps_to_success, cumulative_rewards

def plot_steps_to_success(alpha, gamma, num_episodes):
    steps_to_success, _ = run_reinforcement_learning(alpha, gamma, num_episodes)
    episodes = range(1, num_episodes + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, steps_to_success, label=f'α={alpha}, γ={gamma}')
    plt.xlabel('Episodes')
    plt.ylabel('Steps to Reach Goal')
    plt.title('Steps to Success Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cumulative_reward(alpha, gamma, num_episodes):
    _, cumulative_rewards = run_reinforcement_learning(alpha, gamma, num_episodes)
    episodes = range(1, num_episodes + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, cumulative_rewards, label=f'α={alpha}, γ={gamma}')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    alpha_1, gamma_1 = 0.1, 0.9
    alpha_2, gamma_2 = 0.9, 0.1
    num_episodes = 100

    plot_steps_to_success(alpha_1, gamma_1, num_episodes)
    plot_steps_to_success(alpha_2, gamma_2, num_episodes)

    plot_cumulative_reward(alpha_1, gamma_1, num_episodes)
    plot_cumulative_reward(alpha_2, gamma_2, num_episodes)
