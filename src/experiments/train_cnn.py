import torch
import yaml
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.preprocessing import ArealData
from src.models import EuroSATCNN

class EuroSATExperiment:
    def __init__(self, config_path="config.yaml"):
        # 1. Chargement de la config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Initialisation des modèles et outils
        self.model = EuroSATCNN(
            in_channels=self.config['model']['in_channels'], 
            n_classes=self.config['model']['n_classes']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['model']['learning_rate']
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_val_acc = 0.0
        self.history = {'train_loss': [], 'val_acc': [], 'val_loss': []}

    def _get_loader(self, csv_key, shuffle=False):
        """Méthode interne pour créer les DataLoaders."""
        dataset = ArealData(
            csv_file=self.config['data'][csv_key], 
            root_dir=self.config['data']['root_dir']
        )
        return DataLoader(
            dataset, 
            batch_size=self.config['model']['batch_size'], 
            shuffle=shuffle, 
            num_workers=2
        )

    def run_epoch(self, train_loader, val_loader, epoch):
        # Train
        self.model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{self.config['model']['epochs']}]", unit="batch")
        
        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        val_loss, val_acc = self.evaluate(val_loader, desc="Validation")
        
        # Log History
        self.history['train_loss'].append(avg_train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(self.model.state_dict(), self.config['model']['model_path'])
            print(f"New Best Model Saved ({val_acc:.2f}%)")

    def evaluate(self, loader, desc="Evaluation"):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=desc, leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(loader), 100 * correct / total

    def train(self):
        print(f"--- Starting Experiment on {self.device} ---")
        train_loader = self._get_loader('train_dir', shuffle=True)
        val_loader = self._get_loader('validation_dir', shuffle=False)
        
        for epoch in range(self.config['model']['epochs']):
            self.run_epoch(train_loader, val_loader, epoch)
            
    def test(self):
        print("\n--- Final Test Evaluation ---")
        if os.path.exists(self.config['model']['model_path']):
            self.model.load_state_dict(torch.load(self.config['model']['model_path']))
        
        test_loader = self._get_loader('test_dir', shuffle=False)
        loss, acc = self.evaluate(test_loader, desc="Test Set")
        print(f"Final Results: Test Loss: {loss:.4f} | Test Acc: {acc:.2f}%")
        return acc

if __name__ == "__main__":
    experiment = EuroSATExperiment()
    experiment.train()
    experiment.test()