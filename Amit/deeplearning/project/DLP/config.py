# config.py
class Config:
    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    
    # Model Architecture
    INPUT_SIZE = 784  # 28x28 flattened
    NUM_CLASSES = 10
    
    # Activation Function
    ACTIVATION = "relu"  # e.g., 'relu', 'tanh', 'sigmoid'