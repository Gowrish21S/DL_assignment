Deep Learning & Reinforcement Learning Assignment

This repository showcases five distinct machine learning projects, demonstrating various techniques in Reinforcement Learning, Time Series Forecasting, and Convolutional Neural Networks.

Each section below details the project's purpose and highlights the key modifications made to the original code to enhance performance, stability, or adopt modern best practices.

-----

## 1\. Tic-Tac-Toe Game with Q-Learning

This project implements a self-learning agent for the game of Tic-Tac-Toe using a Reinforcement Learning approach.

### ⚙️ **Key Modification: Transition from State-Value ($V(s)$) to Action-Value ($Q(s, a)$) Learning (SARSA Structure)**

The core change upgraded the agent from a basic Monte Carlo approach to a **Temporal Difference (TD) learning** structure, specifically using a **SARSA-like** update rule.

| Feature | Original (Implicit $V(s)$) | Modified (Explicit $Q(s, a)$) |
| :--- | :--- | :--- |
| **Value Function** | State-Value $V(s)$: Stores the value of being in a state. | **Action-Value $Q(s, a)$:** Stores the value of taking action $a$ in state $s$. |
| **Learning Rule** | Monte Carlo: Updates values only at the *end* of the game, propagating the final reward backward. | **SARSA:** Updates the value *after every move* using the TD error: $Q(S, A) \leftarrow Q(S, A) + \alpha [ R + \gamma \cdot Q(S', A') - Q(S, A) ]$. |
| **Update Timing**| End of episode (Game Over). | **After every state transition** (Move by the opponent). |

### 📈 **Key Modification: Dynamic Exploration Rate ($\epsilon$-Decay)**

A dynamic exploration rate was introduced to manage the exploration-exploitation trade-off during training.

| Feature | Original | Modified |
| :--- | :--- | :--- |
| **Exploration Rate ($\epsilon$)** | Fixed at $\epsilon = 0.3$. | **Starts at $\epsilon = 1.0$ and decays** using a multiplicative factor (e.g., $0.9999$) until a minimum floor ($\epsilon_{min}=0.01$) is reached. |
| **Benefit** | Ensures early wide exploration and later convergence to an optimal, deterministic policy. |

-----

## 2\. AlexNet Architecture Implementation (Keras CNN)

This project implemented and modernized the classical AlexNet Convolutional Neural Network (CNN) for image classification.

### ⚙️ **Key Modification: Architectural Stabilization (Batch Normalization)**

Batch Normalization (BN) layers were strategically added to stabilize and accelerate training.

| Feature | Original AlexNet Code | Modified AlexNet Code |
| :--- | :--- | :--- |
| **Normalization** | None (used Local Response Normalization in the original paper, but omitted here). | **Added `BatchNormalization()`** layer after every `Conv2D` layer and before the first two `Dense` (Fully Connected) layers. |
| **Conv/Dense** | Used `Conv2D(..., activation='relu')`. | Split into `Conv2D(...) -> BatchNormalization() -> Activation('relu')`. |
| **Benefit** | Reduces Internal Covariate Shift, allowing for faster convergence and acting as a mild regularizer. |

### ⚙️ **Key Modification: Keras Best Practice (Explicit Input Layer)**

The model definition was updated to conform to modern Keras best practices, resolving a `UserWarning`.

| Feature | Original AlexNet Code | Modified AlexNet Code |
| :--- | :--- | :--- |
| **Input Definition** | Passed `input_shape` directly to the first `Conv2D` layer. | **Added `Input(shape=input_shape)`** as the very first layer in the `Sequential` model. |
| **Benefit** | Suppresses the Keras `UserWarning` and explicitly defines the input tensor, improving model graph definition reliability. |

-----

## 3\. Time Series Forecasting with LSTM

This project used a Recurrent Neural Network (RNN) to forecast the International Airline Passengers dataset.

### ⚙️ **Key Modification: Increased Depth and Capacity (Stacked LSTMs)**

The model was significantly enhanced by transitioning from a single layer to a deeper, stacked architecture.

| Feature | Original Code (Simple RNN) | Modified Code (Stacked LSTM) |
| :--- | :--- | :--- |
| **Recurrent Layer** | Single `LSTM(10)` layer. | **Two Stacked `LSTM(50)` layers.** |
| **Stacking Requirement** | N/A | First LSTM layer was set to **`return_sequences=True`** to pass sequence output to the second layer. |
| **Capacity** | 10 LSTM units. | **50 LSTM units** per layer. |
| **Training** | 50 epochs. | **100 epochs** (to allow the deeper model to fully converge). |

### ⚙️ **Key Modification: Regularization**

| Feature | Original Code | Modified Code |
| :--- | :--- | :--- |
| **Regularization** | None. | **Added `Dropout(0.2)`** between the two stacked LSTM layers. |
| **Benefit** | Prevents overfitting to the training data, improving generalization to the test set. |

-----

## 4\. Cats and Dogs Image Classification (CNN)

This project focused on binary image classification using a CNN architecture with the Cats vs. Dogs dataset.

### ⚙️ **Key Modification: Data Augmentation**

The most critical change was implementing data augmentation to fight overfitting on the limited dataset.

| Feature | Original Code (`train_datagen`) | Modified Code (`train_datagen`) |
| :--- | :--- | :--- |
| **Data Augmentation** | Only `rescale = 1.0/255.`. | **Added:** `rotation_range=40`, `width_shift_range=0.2`, `zoom_range=0.2`, `horizontal_flip=True`, etc. |
| **Benefit** | Artificially expands the training dataset, significantly improving the model's ability to generalize. |

### ⚙️ **Key Modification: Increased Model Capacity and Dropout**

The base CNN architecture was deepened and widened.

| Feature | Original Code (Basic CNN) | Modified Code (Deeper CNN) |
| :--- | :--- | :--- |
| **Depth** | Three Conv/Pool blocks. | **Four Conv/Pool blocks.** |
| **Filter Count** | Max 64 filters. | **Max 128 filters** in the final Conv layers. |
| **Hidden Layer** | `Dense(512)`. | **`Dense(1024)`**. |
| **Regularization** | No Dropout before Dense. | **Added `Dropout(0.5)`** before the first Dense layer. |

### ⚙️ **Fixes for Google Colab Execution**

  * Corrected the usage of local Windows file paths to use Colab's relative paths (`/content/`).
  * Added the `!unzip` shell command for robust file extraction in the Colab environment.
  * Fixed the `AttributeError` in the feature map visualization by referencing the input via **`model.layers[0].input`**.

-----

## 5\. Character-Level Text Generation (RNN)

This project used a recurrent network for next-character prediction and text generation.

### ⚙️ **Key Modification: Advanced RNN Architecture (LSTM & Stacking)**

The simple `SimpleRNN` layer was replaced with more modern, robust layers.

| Feature | Original Code | Modified Code |
| :--- | :--- | :--- |
| **Recurrent Layer** | `SimpleRNN` (prone to vanishing gradient). | **Two Stacked `LSTM`** layers. |
| **Input Handling** | Sparse, wide **One-Hot Encoding** input. | **Added `Embedding` layer** to map integer input to a dense, trainable vector. |
| **Model Input** | `X_one_hot` (One-Hot). | **$X$ (Integer encoded)**, handled by the Embedding layer. |
| **Regularization** | None. | **Added `Dropout(0.2)`** between LSTM layers. |
| **Benefit** | LSTMs retain long-term dependencies, and the Embedding layer makes the model more efficient and capable of learning semantic relationships between characters. |

-----

## Setup and Execution

### Requirements

To run these projects, you will need the following libraries:

```bash
pip install tensorflow keras pandas numpy matplotlib scikit-learn
```
