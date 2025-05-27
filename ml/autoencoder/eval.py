import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from train import AE

tensor_transform = transforms.ToTensor()
dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=tensor_transform
)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

model = AE()
# model.load_state_dict(torch.load("model_0.pt", map_location=torch.device("cpu")))
model.load_state_dict(torch.load("model_19.pt", map_location=torch.device("cpu")))
model.eval()
dataiter = iter(loader)
images, _ = next(dataiter)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images = images.view(-1, 28 * 28).to(device)
reconstructed = model(images)

fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(10, 10))
for i in range(10):
    image_1 = images[i]
    image_2 = images[i + 10]
    reconstructed_1 = reconstructed[i]
    reconstructed_2 = reconstructed[i + 1]

    latent_1 = model.encoder(image_1)
    latent_2 = model.encoder(image_2)

    # latent_x = latent_1 + latent_2
    l = latent_1.shape[0]   
    latent_x = torch.cat([latent_1[: l // 2], latent_2[l // 2 :]], dim=0)
    reconstructed_x = model.decoder(latent_x)

    axes[i, 0].imshow(image_1.cpu().detach().numpy().reshape(28, 28), cmap="gray")
    axes[i, 0].axis("off")
    axes[i, 1].imshow(image_2.cpu().detach().numpy().reshape(28, 28), cmap="gray")
    axes[i, 1].axis("off")
    axes[i, 2].imshow(reconstructed_x.cpu().detach().numpy().reshape(28, 28), cmap="gray")
    axes[i, 2].axis("off")
plt.show()



# fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 3))
# for i in range(10):
#     axes[0, i].imshow(images[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
#     axes[0, i].axis('off')
#     axes[1, i].imshow(reconstructed[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
#     axes[1, i].axis('off')
# plt.show()
