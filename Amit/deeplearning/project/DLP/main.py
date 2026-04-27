from config import Config
from minist_dataset import MNISTDataset
from neural_network import NeuralNetwork
from trainer import Trainer
from evaluator import Evaluator
import torch.optim as optim
import torch.nn as nn

class TrainingPipeline:
    def run(self):
        print("1. Loading Data...")
        loader = MNISTDataset(Config.BATCH_SIZE).load()
        
        print("2. Building Model...")
        model = NeuralNetwork(Config.INPUT_SIZE, Config.NUM_CLASSES, Config.ACTIVATION)
        
        print("3. Setting up Trainer & Evaluator...")
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        trainer = Trainer(model, optimizer, criterion)
        evaluator = Evaluator() # ضفنا المُمتحن هنا
        
        print("4. Starting Training...")
        for epoch in range(Config.EPOCHS):
            # التدريب
            trainer.train(loader)
            
            # الامتحان وطباعة الدقة بعد كل دورة
            print(f"Epoch {epoch+1}/{Config.EPOCHS} completed! ", end="")
            evaluator.evaluate(model, loader)
            
        print("Training Finished Successfully! 🎉")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()