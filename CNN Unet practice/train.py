import load
import model
from torch.utils.data import DataLoader, random_split
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


class_dict = {
    (0, 0, 0): 0,
    (128, 64, 128): 1,
    (130, 76, 0): 2,
    (0, 102, 0): 3,
    (112, 103, 87): 4,
    (28, 42, 168): 5,
    (48, 41, 30): 6,
    (0, 50, 89): 7,
    (107, 142, 35): 8,
    (70, 70, 70): 9,
    (102, 102, 156): 10,
    (254, 228, 12): 11,
    (254, 148, 12): 12,
    (190, 153, 153): 13,
    (153, 153, 153): 14,
    (255, 22, 96): 15,
    (102, 51, 0): 16,
    (9, 143, 150): 17,
    (119, 11, 32): 18,
    (51, 51, 0): 19,
    (190, 250, 190): 20,
    (112, 150, 146): 21,
    (2, 135, 115): 22,
    (255, 0, 0): 23,
}

def validate(model, loader, loss_fn, device):

    model.eval() # what does this do
    
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():

        for imgs, masks in loader:

            imgs = imgs.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.long)

            preds = model(imgs)
            loss = loss_fn(preds, masks)
            total_loss += loss.item()

            pred_classes = preds.argmax(dim=1)
            correct_pixels += (pred_classes == masks).sum().item()
            total_pixels += masks.numel()

    avg_loss = total_loss / len(loader)
    pixel_acc = correct_pixels / total_pixels

    return avg_loss, pixel_acc



def main():

    split = 0.2

    dataset = load.DroneDataset(
        "semantic_drone_dataset/training_set/images",
        "semantic_drone_dataset/training_set/gt/semantic/label_images",
        class_dict
    )

    val_size = int(split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size = 2, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size = 2, shuffle=False, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    Model = model.UNet(out_ch=24).to(device)

    if os.path.exists("unet.pth"):
        Model.load_state_dict(torch.load("unet.pth", map_location=device))
        print("Resumed from checkpoint")

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        Model.parameters(),
        lr=1e-4
    )

    epochs = 1000

    train_losses = []
    val_losses   = []
    val_accuracies = []

    save_after = 100
    save_png_after = 100
    count = 0


    for e in range(epochs):

        Model.train()

        loop = tqdm(
            train_loader,
            desc=f"Epoch {e+1}/{epochs}",
            colour=("green"),
            ncols=120,
            leave=True,
            ascii=False
        )

        for img, mask in loop:

            imgs = img.to(device=device, dtype=torch.float)

            masks = mask.to(device=device, dtype=torch.long)

            pred = Model(imgs)

            loss = loss_fn(pred, masks)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            count += 1

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}"
            )

            if (not count % save_after) or count == 1:
                train_losses.append(loss.item())

                val_loss, val_acc = validate(Model, val_loader, loss_fn, device)

                Model.train()

                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

                plt.clf()
                plt.plot(train_losses, label="Train Loss", color="blue")
                plt.plot(val_losses,   label="Val Loss",   color="orange")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Train vs Val Loss")
                plt.legend()
                plt.grid(False)

                # Add val loss and accuracy as text in the top-right corner
                textstr = f"Pixel Acc: {val_acc*100:.2f}%"
                plt.gca().text(
                    0.98, 0.95, textstr,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.8)
                )

                plt.pause(0.01)

            if not count % save_png_after:
                plt.savefig("loss_curve.png")

        torch.save(Model.state_dict(), "unet.pth")


if __name__ == "__main__":
    main()