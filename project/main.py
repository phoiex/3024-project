from train import *
from methods import *
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn


criterion = nn.CrossEntropyLoss()
data_loader = SVHNDataLoader(data_dir='./data', batch_size=16)
train_loader, test_loader = data_loader.load_data()

loss_history_train = []
loss_history_test = []
roc_auc_history = []
accuracy_history = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 60
for epoch in range(num_epochs):
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
   
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
  
    average_train_loss = running_loss / len(train_loader)
    loss_history_train.append(average_train_loss)
    print(f"Training Loss: {average_train_loss:.4f}")

    
    test_loss, roc_auc, accuracy = evaluate_model(model, test_loader, device, criterion, loss_history_test, roc_auc_history, accuracy_history)
    

plot_results(loss_history_train, loss_history_test, roc_auc_history, accuracy_history)
evaluate_model_hxjz(model, test_loader, device, criterion)