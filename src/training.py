from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
def train_model(model, optimizer, criterion, X_train, y_train, X_validation, y_validation, epochs=50):
    train_acc = []
    val_acc = []
    train_losses = []
    val_losses = []
    for epoch in tqdm.tqdm(range(epochs), desc="Training Progress"):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train).argmax(dim=1)
            val_preds = model(X_validation).argmax(dim=1)
            train_accuracy = accuracy_score(y_train.cpu(), train_preds.cpu())
            val_accuracy = accuracy_score(y_validation.cpu(), val_preds.cpu())
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)
            train_losses.append(loss.item())
            val_losses.append(criterion(model(X_validation), y_validation).item())

    return train_acc, val_acc, train_losses, val_losses


def plot_learning_curves(train_acc, val_acc, train_losses, val_losses, title="Learning Curves"):
    # save metrics to csv named after title (safe filename)
    try:
        filename = title if title.lower().endswith(".csv") else f"{title}.csv"
        # sanitize filename: keep alnum, space, dot, underscore, dash
        filename = "".join(c if (c.isalnum() or c in (" ", ".", "_", "-")) else "_" for c in filename).strip().replace(" ", "_")

        n = max(len(train_acc), len(val_acc), len(train_losses), len(val_losses))
        def _pad(lst):
            return list(lst) + [np.nan] * (n - len(lst))

        df = pd.DataFrame({
            "epoch": list(range(1, n + 1)),
            "train_acc": _pad(train_acc),
            "val_acc": _pad(val_acc),
            "train_loss": _pad(train_losses),
            "val_loss": _pad(val_losses),
        })

        df.to_csv(filename, index=False)
    except Exception as e:
        print(f"Could not save learning curves to CSV: {e}")
        
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()

def train_complex_model(model, optimizer, criterion, parameters, all_train_dfs, all_test_dfs, label='label', epochs=50, debug=False):
    device = next(model.parameters()).device
    train_acc = []
    val_acc = []
    train_losses = []
    val_losses = []

    # Extract parameters or set defaults
    ma_window = parameters.get('ma_window', 30)
    plateau_threshold = parameters.get('plateau_threshold', 1e-4)
    underfit_threshold = parameters.get('underfit_threshold', 0.4)
    over_fit_threshold_training = parameters.get('over_fit_threshold_training', 0.1)
    over_fit_threshold_validation = parameters.get('over_fit_threshold_validation', 0.1)
    min_neurons = parameters.get('min_neurons', 16)

    current_class_index = 2  # start from the third class since first two are combined initially
    goal_classes = len(all_train_dfs) - 1  # last valid index
  

    if debug:
        print(f"Goal classes to achieve (last index): {goal_classes}")

    def combine_new_class_to_dataset(X_train, y_train, X_val, y_val, new_class_df_train, new_class_df_test):
        # new_class_df_* are expected to have the same columns: [feature1,..,featureN, label]
        if X_train is None or y_train is None:
            combined_train_df = new_class_df_train.copy()
            combined_test_df = new_class_df_test.copy()
        else:
            # determine feature column names from the new class df (exclude label)
            feature_cols = list(new_class_df_train.drop(columns=[label]).columns)

            # move tensors to CPU and convert to numpy safely
            Xtr = X_train.detach().cpu().numpy()
            ytr = y_train.detach().cpu().numpy()
            Xte = X_val.detach().cpu().numpy()
            yte = y_val.detach().cpu().numpy()

            # build DataFrames that match the same column layout as new_class_df_train
            train_features = pd.DataFrame(Xtr, columns=feature_cols)
            train_labels = pd.Series(ytr.flatten(), name=label)
            test_features = pd.DataFrame(Xte, columns=feature_cols)
            test_labels = pd.Series(yte.flatten(), name=label)

            existing_train_df = pd.concat([train_features, train_labels], axis=1)
            existing_test_df = pd.concat([test_features, test_labels], axis=1)

            # now append new class rows (axis=0 = rows) and shuffle
            combined_train_df = pd.concat([existing_train_df, new_class_df_train], axis=0).sample(frac=1).reset_index(drop=True)
            combined_test_df = pd.concat([existing_test_df, new_class_df_test], axis=0).sample(frac=1).reset_index(drop=True)



        # convert back to tensors with consistent column ordering
        X_train = torch.tensor(combined_train_df.drop(columns=[label]).values, dtype=torch.float32).to(device)
        y_train = torch.tensor(combined_train_df[label].values, dtype=torch.long).to(device)
        X_val = torch.tensor(combined_test_df.drop(columns=[label]).values, dtype=torch.float32).to(device)
        y_val = torch.tensor(combined_test_df[label].values, dtype=torch.long).to(device)

        return X_train, y_train, X_val, y_val

    combined_train_df = pd.DataFrame()
    combined_test_df = pd.DataFrame()
    X_train, y_train, X_val, y_val = None, None, None, None
    X_train, y_train, X_val, y_val = combine_new_class_to_dataset(X_train, y_train, X_val, y_val, pd.DataFrame(all_train_dfs[0]), pd.DataFrame(all_test_dfs[0]))
    X_train, y_train, X_val, y_val = combine_new_class_to_dataset(X_train, y_train, X_val, y_val, pd.DataFrame(all_train_dfs[1]), pd.DataFrame(all_test_dfs[1]))

    if debug:
        print(f"Initial training with classes: {[df[label].iloc[0] for df in all_train_dfs[:2]]}, Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")
    epoch_counter = -1

    # for epoch in tqdm.tqdm(range(epochs), desc="Training Progress"):
    for epoch in range(epochs):
        epoch_counter += 1
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # compute and store losses and accuracies for diagnostics
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train).argmax(dim=1)
            val_preds = model(X_val).argmax(dim=1)
            train_accuracy = accuracy_score(y_train.cpu(), train_preds.cpu())
            val_accuracy = accuracy_score(y_val.cpu(), val_preds.cpu())
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)
            # losses
            train_loss = loss.item()
            val_loss = criterion(model(X_val), y_val).item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        lookback = min(ma_window, epoch_counter + 1)
        lookback = max(1, int(lookback))

        if debug:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        if len(train_losses) >= lookback:
            # compute moving averages
            ma_train = np.convolve(train_losses, np.ones(lookback)/lookback, mode='valid')
            ma_val = np.convolve(val_losses, np.ones(lookback)/lookback, mode='valid')

            # compute trend (slope) over the last ma_window MA points using a linear fit
            n_train = min(len(ma_train), lookback)
            n_val = min(len(ma_val), lookback)

            if n_train >= 2:
                x = np.arange(n_train, dtype=np.float64)
                y = ma_train[-n_train:].astype(np.float64)
                train_slope = float(np.polyfit(x, y, 1)[0])
            else:
                train_slope = 0.0

            if n_val >= 2:
                x = np.arange(n_val, dtype=np.float64)
                y = ma_val[-n_val:].astype(np.float64)
                val_slope = float(np.polyfit(x, y, 1)[0])
            else:
                val_slope = 0.0

            # plateau detection: both losses stop decreasing significantly
            is_plateaued = abs(train_slope) < plateau_threshold and abs(val_slope) < plateau_threshold
            
            if is_plateaued and debug:
                print(f"[train] Plateau detected at epoch {epoch}. train_slope={train_slope:.6f}, val_slope={val_slope:.6f}")
                if current_class_index > goal_classes:
                    print("All classes have been added. Continuing training on final dataset.")

            # UNDERFITTING (loss-based): both losses are high and decreasing slowly
            if (train_slope > -plateau_threshold and val_slope > -plateau_threshold) and (ma_train[-1] > underfit_threshold):
                if debug:
                    plateau_msg = " (during plateau)" if is_plateaued else ""
                    print(f"[train] Underfitting detected at epoch {epoch}{plateau_msg}. train_loss={ma_train[-1]:.4f}, val_loss={ma_val[-1]:.4f}")
                changed = model.add_hidden_neurons()
                if changed:
                    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])
                    if debug:
                        print(f"[train] Model structure changed: added neurons. Hidden layers: {len(model.hidden_layers)}")

            # OVERFITTING (loss-based): train loss keeps decreasing but val loss increases (gap grows)
            elif (train_slope < -over_fit_threshold_training and val_slope > over_fit_threshold_validation) or train_accuracy > 0.99:
                if debug:
                    plateau_msg = " (during plateau)" if is_plateaued else ""
                    print(f"[train] Overfitting detected at epoch {epoch}{plateau_msg}. train_loss={ma_train[-1]:.4f}, val_loss={ma_val[-1]:.4f}")
                if isinstance(model.hidden_layers[0], nn.Identity) or model.hidden_layers[0].out_features < min_neurons:
                    if debug:
                        print("[train] Adding neurons to prevent overfitting.")
                    model.add_hidden_neurons()
                    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])
                elif current_class_index <= goal_classes:
                    if debug:
                        print(f"Adding new class {current_class_index} to the dataset.")
                    X_train, y_train, X_val, y_val = combine_new_class_to_dataset(X_train, y_train, X_val, y_val, all_train_dfs[current_class_index], all_test_dfs[current_class_index])
                    model.update_output_layer(len(torch.unique(y_train)))
                    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])
                    current_class_index += 1
                else:
                    if debug:
                        print("All classes have been added. Cannot add more classes. Continuing training.")
                    break
            
            elif is_plateaued:
                # This handles cases where we've plateaued but don't meet strict overfitting/underfitting criteria
                if current_class_index <= goal_classes:
                    if debug:
                        print(f"[train] Plateau without clear over/underfitting. Adding class {current_class_index} to continue progress.")
                    X_train, y_train, X_val, y_val = combine_new_class_to_dataset(X_train, y_train, X_val, y_val, all_train_dfs[current_class_index], all_test_dfs[current_class_index])
                    model.update_output_layer(len(torch.unique(y_train)))
                    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])
                    current_class_index += 1
                elif debug:
                    print("[train] Plateau detected but no action needed - all classes added and no clear over/underfitting.")


    print(f"Parameters used: {parameters}")
    print(f"Training finished. Reached class index: {current_class_index}. Final train acc: {train_acc[-1]:.4f}, test acc: {val_acc[-1]:.4f}")
    if current_class_index < goal_classes:
        return [0], [0], [0], [0]
    
    return train_acc, val_acc, train_losses, val_losses
