import torch
import os
import models
import argparse
from utils.utils import get_dataloaders, save_to_csv
from utils.masks import StaticMaskedDataset, glimpse_mask
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> USING DEVICE: {DEVICE}")

# --- CPU OPTIMIZATION ---
cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
torch.set_num_threads(cpus_per_task)
print(f">>> TORCH THREADS: {torch.get_num_threads()}")
MODEL_CLASS = models.ModeratelySmallCNN

# -- EXPERIMENT CONFIG ---
NUM_EPOCHS: int=200
BATCH_SIZE: int=32
LEARNING_RATE: float=0.001
GRID_SIZE: int=77
EPOCH_COUNTER: int=25

# --- TRAIN ---

def train(num_glimpses: int, patch_size: int, loaders: tuple[DataLoader, DataLoader]) -> tuple[float, dict[str, list[float]]]:
    """
    Executes training loop with dynamic masking and validation loop with static masking.
    Returns:
        best_val_acc (float): Best validation accuracy.
        history (dict): Dictionary containing epoch-wise metrics.
    """
    train_loader, val_loader = loaders

    model = MODEL_CLASS().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    model_name = model.__class__.__name__
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> MODEL: {model_name} ({total_params} PARAMETERS)")

    # Per-epoch metrics
    history: dict[str, list[float]] = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "val_f1": []
    }
    best_val_acc: float = 0.0

    # Lists for calculating interval averages
    interval_val_accs: list[float] = []
    interval_train_accs: list[float] = []
    interval_val_losses: list[float] = []
    interval_train_losses: list[float] = []

    # tqdm logging
    format_str = "{desc}: {percentage:3.0f}% |{bar:25}| {n_fmt}/{total_fmt} [Elapsed: {elapsed} | Remaining: {remaining}] {postfix}"
    pbar = tqdm(range(NUM_EPOCHS), desc="Training", bar_format=format_str, unit="epoch", ncols=150, mininterval=5.0)

    for epoch in pbar:
        # --- TRAINING PHASE ---
        model.train()
        train_correct, train_total, running_train_loss = 0, 0, 0.0

        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            inputs = glimpse_mask(images, num_glimpses, patch_size)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = 100.0 * (train_correct / train_total)

        # --- VALIDATION PHASE ---
        model.eval()
        val_correct, val_total, running_val_loss = 0, 0, 0.0
        all_preds, all_targets = [] , []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                logits = model(inputs)
                loss = criterion(logits, targets)
                running_val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = 100.0 * (val_correct / val_total)
        epoch_f1 = float(f1_score(all_targets, all_preds))

        # Update history
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_f1"].append(epoch_f1)

        # Track interval and best accuracy
        interval_val_accs.append(epoch_val_acc)
        interval_train_accs.append(epoch_train_acc)
        interval_val_losses.append(epoch_val_loss)
        interval_train_losses.append(epoch_train_loss)
        best_val_acc = max(best_val_acc, epoch_val_acc)
        
        # --- LOGGING ---
        pbar.set_postfix({
            "T_Loss": f"{epoch_train_loss:.3f}",
            "V_Loss": f"{epoch_val_loss:.3f}",
            "T_Acc": f"{epoch_train_acc:.1f}%",
            "V_Acc": f"{epoch_val_acc:.1f}%",
            "V_F1": f"{epoch_f1:.3f}"
        })
        
        if (epoch + 1) % EPOCH_COUNTER == 0:
            avg_val_acc = sum(interval_val_accs) / len(interval_val_accs)
            avg_train_acc = sum(interval_train_accs) / len(interval_train_accs)
            avg_train_loss = sum(interval_train_losses) / len(interval_train_losses)
            avg_val_loss = sum(interval_val_losses) / len(interval_val_losses)

            summary = (
                f"\n>>> Interval: (Epochs {epoch+2-EPOCH_COUNTER}-{epoch+1})\n"
                f">>> Avg Loss: T:{avg_train_loss:.4f} V:{avg_val_loss:.4f}\n"
                f">>> Avg Acc:  T:{avg_train_acc:5.2f}% V:{avg_val_acc:5.2f}%\n"
                f">>> Best Val Acc: {best_val_acc:5.2f}%"
            )

            tqdm.write(summary)

            # Reset interval lists
            interval_val_accs, interval_train_accs = [], []
            interval_val_losses, interval_train_losses = [], []

    return best_val_acc, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Active Sensing Model")
    parser.add_argument("--seed", type=int, help="Random seed for the experiment")
    parser.add_argument("--glimpses", type=int, nargs="+", help="List of glimpse counts to sweep (e.g. 14 16 20 25)")
    parser.add_argument("--patch_size", type=int, default=3, help="Side length of the square patch")
    parser.add_argument("--results_file", type=str, default="results_moderatelysmall.csv", help="Filename for CSV results")
    args = parser.parse_args()

    # Determine execution mode
    seeds = [args.seed] if args.seed is not None else [1]
    num_glimpses_list = args.glimpses if args.glimpses is not None else [14, 16, 20, 25]
    patch_sizes = [args.patch_size] 

    for patch_size in patch_sizes:
        for seed in seeds:
            print(f">>> PATCH SIZE: {patch_size} | SEED: {seed}")
            train_loader, val_loader = get_dataloaders(grid_size=GRID_SIZE, batch_size=BATCH_SIZE, seed=seed)
            if train_loader is None:
                break

            for n in num_glimpses_list:
                print(f">>> NUMBER OF GLIMPSES: {n}")
                # wrap *only* val set for static evaluation
                static_dataset = StaticMaskedDataset(val_loader.dataset, n, patch_size, seed)
                static_loader = DataLoader(static_dataset, batch_size=BATCH_SIZE, shuffle=False)

                # train using base_train_loader (dynamic) and val_loader (static)
                best_val_acc, history = train(n, patch_size, (train_loader, static_loader))

                # Log results
                rows = [{
                    "patch_size": patch_size,
                    "glimpses": n,
                    "seed": seed,
                    "epoch": i + 1,
                    "val_accuracy": history['val_acc'][i],
                    "train_accuracy": history['train_acc'][i],
                    "val_loss": history['val_loss'][i],
                    "train_loss": history['train_loss'][i],
                    "val_f1": history['val_f1'][i],
                    "best_val_accuracy": best_val_acc
                } for i in range(len(history['val_acc']))]

                save_to_csv(args.results_file, rows)

    print(f"\nData saved to {args.results_file}")

