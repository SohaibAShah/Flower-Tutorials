"""up-fall-app: A Flower / PyTorch app."""
from collections import OrderedDict   
import random
import numpy as np


def pareto_optimization(rf_loss, rf_acc_train, rf_acc_val, rf_acc_global, p_loss, p_bias, client_num, client_ids):
    """Implements Pareto optimization to select clients."""
    # Convert metric dicts to a numpy array, ensuring a consistent order via client_ids
    data_points = [
        np.array([rf_loss.get(cid, 0), rf_acc_train.get(cid, 0), rf_acc_val.get(cid, 0),
                  rf_acc_global.get(cid, 0), -p_loss.get(cid, 0), -p_bias.get(cid, 0)])
        for cid in client_ids
    ]
    data = np.array(data_points)

    print(f"Constructed data matrix for Pareto: shape={data.shape}")

    # Pareto front selection
    def is_dominated(point, others):
        """判断 point 是否被 others 支配"""
        return any(np.all(other >= point) and np.any(other > point) for other in others)

    pareto_indices = [
        i for i, point in enumerate(data) if not is_dominated(point, np.delete(data, i, axis=0))
    ]
    pareto_clients = pareto_indices
    print(f"Pareto front client indices: {pareto_clients}")

    # If more Pareto clients than needed, randomly select
    if len(pareto_clients) > client_num:
        selected = random.sample(pareto_clients, client_num)
        print(f"More Pareto clients than needed. Randomly selected: {selected}")
        print("=== Pareto Optimization: End ===")
        return [int(x) for x in selected]

    # If fewer Pareto clients, fill with best scores
    remaining_slots = client_num - len(pareto_clients)
    pareto_scores = [0.4 * rf_loss[i] + 0.6 * rf_acc_global[i] for i in range(len(rf_loss))]
    print("Pareto scores for all clients:", [f"{x:.2f}" for x in pareto_scores])
    sorted_indices = np.argsort(pareto_scores)[::-1]  # Descending order
    print("Sorted indices by Pareto score:", [int(x) for x in sorted_indices])

    selected_clients = set(pareto_clients)
    print(f"Initial selected clients (Pareto front): {[int(x) for x in selected_clients]}")
    for i in sorted_indices:
        if len(selected_clients) >= client_num:
            break
        if i not in selected_clients:
            selected_clients.add(int(i))
            print(f"Added client {int(i)} to fill remaining slots.")
            # If we have filled all slots, we can stop
            if len(selected_clients) >= client_num:
                break

    print(f"Final selected clients: {[int(x) for x in selected_clients]}")
    print("=== Pareto Optimization: End ===")
    return [int(x) for x in selected_clients]


# --- Replace ALL old metric helpers with these dictionary-based versions ---

def calculate_relative_loss_reduction_as_list(client_losses):
    """Calculates RF_loss. Returns a DICTIONARY {cid: score}."""
    loss_reductions = {}
    for cid, losses in client_losses.items():
        if losses and len(losses) >= 2:
            loss_reductions[cid] = losses[0] - losses[-1]

    if not loss_reductions: return {cid: 0.0 for cid in client_losses.keys()}
    max_loss_reduction = max(loss_reductions.values())
    if max_loss_reduction == 0: return {cid: 0.0 for cid in client_losses.keys()}
    
    return {cid: loss_reductions.get(cid, 0.0) / max_loss_reduction for cid in client_losses.keys()}

def calculate_relative_train_accuracy(client_acc):
    """Calculates RF_ACC_Train. Returns a DICTIONARY {cid: score}."""
    if not client_acc: return {}
    max_acc = max(client_acc.values())
    if max_acc == 0: return {cid: 0.0 for cid in client_acc.keys()}
    return {cid: acc / max_acc for cid, acc in client_acc.items()}

def calculate_global_validation_accuracy(train_acc, global_acc):
    """Calculates RF_ACC_Global. Returns a DICTIONARY {cid: score}."""
    if not train_acc or not global_acc: return {}
    max_global_acc = max(global_acc.values()) if global_acc else 0
    if max_global_acc == 0: max_global_acc = 1.0

    global_train_diff = {cid: global_acc.get(cid, 0) - train_acc.get(cid, 0) for cid in train_acc.keys()}
    max_global_train_diff = max(global_train_diff.values()) if global_train_diff else 0
    if max_global_train_diff == 0: max_global_train_diff = 1.0
    
    return {cid: (global_acc.get(cid, 0) / max_global_acc) - (diff / max_global_train_diff) for cid, diff in global_train_diff.items()}

def calculate_loss_outliers(client_losses, lambda_loss=1.5):
    """Calculates P_loss. Returns a DICTIONARY {cid: score}."""
    final_losses = {cid: losses[-1] for cid, losses in client_losses.items() if losses}
    if not final_losses: return {cid: 0.0 for cid in client_losses.keys()}

    loss_values = np.array(list(final_losses.values()))
    mean_loss, std_loss = np.mean(loss_values), np.std(loss_values)
    threshold = mean_loss + lambda_loss * std_loss
    max_loss = np.max(loss_values)
    if max_loss == 0: return {cid: 0.0 for cid in client_losses.keys()}
    
    all_client_scores = {}
    for cid in client_losses.keys():
        final_loss = final_losses.get(cid, 0.0)
        score = final_loss / max_loss if final_loss > threshold else 0.0
        all_client_scores[cid] = score
    return all_client_scores

def calculate_performance_bias(val_acc, global_acc):
    """Calculates P_bias. Returns a DICTIONARY {cid: score}."""
    if not val_acc: return {}
    
    bias_dict = {}
    for cid, val in val_acc.items():
        global_val = global_acc.get(cid, 0.0)
        max_val = max(val, global_val)
        bias = 0.0 if max_val == 0 else abs(val - global_val) / max_val
        bias_dict[cid] = bias
    return bias_dict