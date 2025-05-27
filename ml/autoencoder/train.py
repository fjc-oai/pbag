"""
Epoch 1/20, Loss: 0.037721
Epoch 2/20, Loss: 0.034123
Epoch 3/20, Loss: 0.034046
Epoch 4/20, Loss: 0.027511
Epoch 5/20, Loss: 0.027402
Epoch 6/20, Loss: 0.025731
Epoch 7/20, Loss: 0.019124
Epoch 8/20, Loss: 0.024532
Epoch 9/20, Loss: 0.021455
Epoch 10/20, Loss: 0.018222
Epoch 11/20, Loss: 0.018332
Epoch 12/20, Loss: 0.021090
Epoch 13/20, Loss: 0.020436
Epoch 14/20, Loss: 0.019164
Epoch 15/20, Loss: 0.020438
Epoch 16/20, Loss: 0.020274
Epoch 17/20, Loss: 0.021409
Epoch 18/20, Loss: 0.025115
Epoch 19/20, Loss: 0.018275
Epoch 20/20, Loss: 0.019591
"""
import os
import urllib.request

import torch

USE_PROXY = False

if USE_PROXY:
    proxy = "http://localhost:3128"
    os.environ.setdefault("HTTP_PROXY", proxy)
    os.environ.setdefault("HTTPS_PROXY", proxy)

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    )
    urllib.request.install_opener(opener)
from torchvision import datasets, transforms


class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def main():
    tensor_transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=tensor_transform
    )
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    model = AE()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    epochs = 20
    outputs = []
    losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        for images, _ in loader:
            images = images.view(-1, 28 * 28).to(device)

            reconstructed = model(images)
            loss = loss_function(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        outputs.append((epoch, images, reconstructed))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

        torch.cuda.synchronize()
        torch.save(model.state_dict(), f"model_{epoch}.pt")


if __name__ == "__main__":
    main()