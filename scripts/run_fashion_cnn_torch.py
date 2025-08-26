import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np, os

RNG = 42
BATCH = 128
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tfm = transforms.Compose([transforms.ToTensor()])  # [0,1]
train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)
test_ds  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32*7*7, 10)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.max_pool2d(x, 2)     # 28->14
        x = F.relu(self.c2(x)); x = F.max_pool2d(x, 2)     # 14->7
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_one_epoch(net, opt, dl):
    net.train(); tot=0; correct=0; loss_sum=0.0
    for x,y in dl:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = net(x); loss = F.cross_entropy(out, y)
        loss.backward(); opt.step()
        loss_sum += loss.item()*x.size(0)
        pred = out.argmax(1); correct += (pred==y).sum().item(); tot += x.size(0)
    return loss_sum/tot, correct/tot

@torch.no_grad()
def evaluate(net, dl):
    net.eval(); tot=0; correct=0; loss_sum=0.0
    for x,y in dl:
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = net(x); loss = F.cross_entropy(out, y)
        loss_sum += loss.item()*x.size(0)
        pred = out.argmax(1); correct += (pred==y).sum().item(); tot += x.size(0)
    return loss_sum/tot, correct/tot

def main():
    torch.manual_seed(RNG)
    net = SmallCNN().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    print(f"Device: {DEVICE}")
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(net, opt, train_dl)
        te_loss, te_acc = evaluate(net, test_dl)
        print(f"Epoch {ep}: train acc={tr_acc:.3f} loss={tr_loss:.3f} | test acc={te_acc:.3f} loss={te_loss:.3f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(net.state_dict(), "outputs/fashion_cnn.pt")
    print("Saved → outputs/fashion_cnn.pt")

if __name__ == "__main__":
    main()
