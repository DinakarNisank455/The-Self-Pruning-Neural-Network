import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ---------------------------
# Prunable Linear Layer
# ---------------------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def sparsity_loss(self):
        gates = torch.sigmoid(self.gate_scores)
        return torch.sum(gates)


# ---------------------------
# Neural Network
# ---------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def total_sparsity_loss(self):
        return (
            self.fc1.sparsity_loss() +
            self.fc2.sparsity_loss() +
            self.fc3.sparsity_loss()
        )


# ---------------------------
# Training Setup
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ---------------------------
# Training Loop
# ---------------------------
def train(model, lambda_val=1e-5, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            sparsity = model.total_sparsity_loss()

            loss = ce_loss + lambda_val * sparsity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# ---------------------------
# Evaluation
# ---------------------------
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# ---------------------------
# Sparsity Calculation
# ---------------------------
def calculate_sparsity(model, threshold=1e-2):
    total = 0
    zero = 0

    for layer in [model.fc1, model.fc2, model.fc3]:
        gates = torch.sigmoid(layer.gate_scores)
        total += gates.numel()
        zero += (gates < threshold).sum().item()

    return 100 * zero / total


# ---------------------------
# Run Experiments
# ---------------------------
lambdas = [1e-6, 1e-5, 1e-4]

results = []

for lam in lambdas:
    print(f"\nTraining with lambda = {lam}")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, lambda_val=lam)
    acc = evaluate(model)
    sparsity = calculate_sparsity(model)

    results.append((lam, acc, sparsity))
    print(f"Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")

print("\nFinal Results:")
for r in results:
    print(f"Lambda: {r[0]}, Accuracy: {r[1]:.2f}%, Sparsity: {r[2]:.2f}%")