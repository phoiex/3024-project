import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns

def train_model(model, train_loader, criterion, optimizer, device, loss_history):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    loss_history.append(average_loss)
    print(f"Training Loss: {average_loss}")

def evaluate_model(model, test_loader, device, criterion, loss_history, roc_auc_history, accuracy_history):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_outputs = []
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

    average_test_loss = test_loss / len(test_loader)
    loss_history.append(average_test_loss)
    print(f"Test Loss: {average_test_loss}")

    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    
    roc_auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr')
    roc_auc_history.append(roc_auc)
    accuracy_history.append(accuracy)

    print(f"ROC AUC: {roc_auc}")

    
    return average_test_loss, roc_auc, accuracy

def plot_results(loss_history_train, loss_history_test, roc_auc_history, accuracy_history):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(loss_history_train, label='Train Loss')
    plt.plot(loss_history_test, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    
    plt.subplot(2, 1, 2)
    epochs = range(1, len(roc_auc_history) + 1)
    plt.plot(epochs, roc_auc_history, label='ROC AUC', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC Curve')
    plt.legend()

    plt.tight_layout()  
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(epochs, accuracy_history, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_model_hxjz(model, test_loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_outputs = []
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, np.argmax(all_outputs, axis=1))
    print("Confusion Matrix:")
    print(conf_matrix)


    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    

