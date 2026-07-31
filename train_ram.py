import torch
import os
import copy
import argparse
import models
from utils.utils import get_dataloaders, save_to_csv, save_weights_safetensors, set_seed
from tqdm import tqdm
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> USING DEVICE: {DEVICE}")

# --- CPU OPTIMIZATION ---
slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if slurm_cpus:
    cpus_per_task = int(slurm_cpus)
    # Use 0 workers if only 1 CPU to avoid context switching overhead
    num_workers = cpus_per_task if cpus_per_task > 1 else 0
else:
    cpus_per_task = os.cpu_count() or 1
    num_workers = 0

torch.set_num_threads(cpus_per_task)
print(f">>> TORCH THREADS: {torch.get_num_threads()} | WORKERS: {num_workers}")

MODEL_CLASS = models.RecurrentAttentionModelClassic

# -- EXPERIMENT CONFIG ---
NUM_EPOCHS: int=200
BATCH_SIZE: int=32
LEARNING_RATE: float=0.001
GRID_SIZE: int=77
EPOCH_COUNTER: int=25

# --- TRAIN ---

def train(num_glimpses: int, patch_size: int, std: float, loaders: tuple, random_baseline: bool = False) -> tuple:
    """
    Executes training loop with REINFORCE and dynamic masking.
    Returns:
        model (nn.Module): The trained model.
        best_val_acc (float): Best validation accuracy.
        history (dict): Dictionary containing epoch-wise metrics.
    """
    train_loader, val_loader = loaders

    model = MODEL_CLASS(patch_size=patch_size, std=std).to(DEVICE)

    # In more complex implementations, we might use a separate optimizer/LR for the baseline net.
    # For simplicity, we use one optimizer for all parameters.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = torch.nn.BCEWithLogitsLoss()
    mse_criterion = torch.nn.MSELoss(reduction='none')

    model_name = model.__class__.__name__
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> MODEL: {model_name} ({total_params} PARAMETERS)")

    # Per-epoch metrics
    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "val_f1": []
    }
    best_val_acc: float = -1.0
    best_state_dict = copy.deepcopy(model.state_dict())

    # Lists for calculating interval averages
    interval_val_accs = []
    interval_train_accs = []
    interval_val_losses = []
    interval_train_losses = []

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

            logits, log_pis, baselines, locations = model(images, num_glimpses, patch_size, random_baseline=random_baseline)

            # 1. Classification loss
            bce_loss = criterion(logits.view(-1), targets.view(-1))

            # Predict
            preds = (torch.sigmoid(logits) > 0.5).float()

            # 2. Reward: 1 if correct, 0 if incorrect
            # We reshape targets and preds to match so we can squeeze to get a 1D tensor
            rewards = (preds.view(-1) == targets.view(-1)).float() # (N,)

            if not random_baseline:
                # Compute policy loss and baseline loss
                policy_loss = 0.0
                baseline_loss = 0.0

                # log_pis is empty for a 1-glimpse model (zero location actions taken)
                if len(log_pis) > 0:
                    for log_pi, b_t in zip(log_pis, baselines):
                        # Baseline loss: MSE between predicted baseline and actual reward
                        baseline_loss += mse_criterion(b_t, rewards).mean()

                        # Policy loss: -log_pi * (R - b_t)
                        # b_t must be detached so gradients don't flow back through baseline net from policy loss
                        advantage = rewards - b_t.detach()
                        policy_loss += (-log_pi * advantage).mean()

                    # Average over T-1 actions actually taken
                    policy_loss = policy_loss / len(log_pis)
                    baseline_loss = baseline_loss / len(baselines)

                loss = bce_loss + policy_loss + baseline_loss
            else:
                loss = bce_loss
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_correct += (preds.view(-1) == targets.view(-1)).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = 100.0 * (train_correct / train_total)

        # --- VALIDATION PHASE ---
        model.eval()
        val_correct, val_total, running_val_loss = 0, 0, 0.0
        all_preds, all_targets = [] , []

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)

                logits, log_pis, baselines, locations = model(images, num_glimpses, patch_size, random_baseline=random_baseline)

                bce_loss = criterion(logits.view(-1), targets.view(-1))

                preds = (torch.sigmoid(logits) > 0.5).float()
                rewards = (preds.view(-1) == targets.view(-1)).float()

                if not random_baseline:
                    policy_loss = 0.0
                    baseline_loss = 0.0

                    # Note: during eval, log_pi is 0, so policy_loss is 0, but we compute it for logging parity
                    # log_pis is empty for a 1-glimpse model
                    if len(log_pis) > 0:
                        for log_pi, b_t in zip(log_pis, baselines):
                            baseline_loss += mse_criterion(b_t, rewards).mean()
                            advantage = rewards - b_t.detach()
                            policy_loss += (-log_pi * advantage).mean()

                        policy_loss = policy_loss / len(log_pis)
                        baseline_loss = baseline_loss / len(baselines)

                    loss = bce_loss + policy_loss + baseline_loss
                else:
                    loss = bce_loss

                running_val_loss += loss.item()
                val_correct += (preds.view(-1) == targets.view(-1)).sum().item()
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
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state_dict = copy.deepcopy(model.state_dict())

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

    model.load_state_dict(best_state_dict)
    return model, best_val_acc, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RAM Model")
    parser.add_argument("--seed", type=int, help="Random seed for the experiment")
    parser.add_argument("--glimpses", type=int, nargs="+", help="List of glimpse counts to sweep (e.g. 14 16 20 25)")
    parser.add_argument("--patch_size", type=int, default=8, help="Side length of the square patch")
    parser.add_argument("--data_dir", type=str, help="Directory containing the datasets")
    parser.add_argument("--output_dir", type=str, help="Directory to save CSV results")
    parser.add_argument("--std", type=float, default=0.1, help="Standard deviation for location policy")
    parser.add_argument("--baseline_lr", type=float, default=0.001, help="Learning rate for baseline network (not currently used separately)")
    parser.add_argument("--random_baseline", action="store_true", help="Use uniform random glimpse locations instead of the learned policy")
    args = parser.parse_args()

    # Determine execution mode
    seeds = [args.seed] if args.seed is not None else [1]
    num_glimpses_list = args.glimpses if args.glimpses is not None else [14, 16, 20, 25]
    patch_sizes = [args.patch_size]

    # Task ID for filename resolution
    file_id = args.seed if args.seed is not None else os.environ.get("SLURM_ARRAY_TASK_ID", 1)
    model_tag = "random" if args.random_baseline else "policy"
    results_file = f"results_ram_{model_tag}_seed_{file_id}.csv"

    # Directory resolution logic
    data_dir = args.data_dir or os.environ.get("SLURM_TMPDIR") or os.environ.get("DATASET_ROOT") or "./data"
    output_dir = args.output_dir or os.environ.get("OUTPUT_DIR") or "./results"

    for patch_size in patch_sizes:
        for seed in seeds:
            set_seed(seed)
            print(f">>> PATCH SIZE: {patch_size} | SEED: {seed}")
            train_loader, val_loader = get_dataloaders(data_dir=data_dir, grid_size=GRID_SIZE, batch_size=BATCH_SIZE, seed=seed)
            if train_loader is None:
                break

            for n in num_glimpses_list:
                print(f">>> NUMBER OF GLIMPSES: {n}")

                # Unlike train.py, we don't wrap val_loader in StaticMaskedDataset
                # because the model determines its own sequence of glimpses dynamically.

                model, best_val_acc, history = train(n, patch_size, args.std, (train_loader, val_loader), random_baseline=args.random_baseline)

                # Save trained weights in safetensors format
                weight_metadata = {
                    "model_class": model.__class__.__name__,
                    "model_type": model_tag,
                    "patch_size": str(patch_size),
                    "num_glimpses": str(n),
                    "seed": str(seed),
                    "std": str(args.std),
                    "num_epochs": str(NUM_EPOCHS),
                    "best_val_accuracy": f"{best_val_acc:.2f}",
                    "format": "pytorch",
                }
                save_weights_safetensors(
                    model,
                    os.path.join(output_dir, "model.safetensors"),
                    metadata=weight_metadata,
                )

                # Log results
                rows = [{
                    "patch_size": patch_size,
                    "glimpses": n,
                    "seed": seed,
                    "model_type": model_tag,
                    "epoch": i + 1,
                    "val_accuracy": history['val_acc'][i],
                    "train_accuracy": history['train_acc'][i],
                    "val_loss": history['val_loss'][i],
                    "train_loss": history['train_loss'][i],
                    "val_f1": history['val_f1'][i],
                    "best_val_accuracy": best_val_acc
                } for i in range(len(history['val_acc']))]

                save_to_csv(output_dir, results_file, rows)

    print(f"\nData saved to {os.path.join(output_dir, results_file)}")
