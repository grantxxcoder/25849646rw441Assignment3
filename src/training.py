from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def train_model(model, optimizer, criterion, X_train, y_train, X_validation, y_validation, epochs=50):
    train_acc = []
    val_acc = []
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
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

def train_complex_model(model, optimizer, criterion, parameters, all_train_dfs, all_test_dfs, label='label', epochs=50):
    device = next(model.parameters()).device
    train_acc = []
    test_acc = []
    train_losses = []
    val_losses = []
    look_back = 30

    has_underfit = False
    has_overfit = False
    has_plateaued = False
    lock = False
    lock_epochs = 0

    X_train, y_train, X_test, y_test = None, None, None, None
    current_class_index = 2  # start from the third class since first two are combined initially
    num_plateued = 0
    goal_classes = len(all_train_dfs) - 1  # last valid index
    final_dataset_reached = False
    final_plateau_count = 0
    print(f"Goal classes to achieve (last index): {goal_classes}")

    def combine_new_class_to_dataset(new_class_df_train, new_class_df_test):
        nonlocal X_train, y_train, X_test, y_test
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
            Xte = X_test.detach().cpu().numpy()
            yte = y_test.detach().cpu().numpy()

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
        X_test = torch.tensor(combined_test_df.drop(columns=[label]).values, dtype=torch.float32).to(device)
        y_test = torch.tensor(combined_test_df[label].values, dtype=torch.long).to(device)

    combined_train_df = pd.DataFrame()
    combined_test_df = pd.DataFrame()
    combine_new_class_to_dataset(pd.DataFrame(all_train_dfs[0]), pd.DataFrame(all_test_dfs[0]))
    combine_new_class_to_dataset(pd.DataFrame(all_train_dfs[1]), pd.DataFrame(all_test_dfs[1]))

    # X_train = torch.tensor(combined_train_df.drop(columns=[label]).values, dtype=torch.float32).to(device)
    # y_train = torch.tensor(combined_train_df[label].values, dtype=torch.long).to(device)

    # X_test = torch.tensor(combined_test_df.drop(columns=[label]).values, dtype=torch.float32).to(device)
    # y_test = torch.tensor(combined_test_df[label].values, dtype=torch.long).to(device)
    print(f"Initial training with classes: {[df[label].iloc[0] for df in all_train_dfs[:2]]}, Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    epoch = 0

    while epoch < epochs:
        epoch += 1
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
            test_preds = model(X_test).argmax(dim=1)
            train_accuracy = accuracy_score(y_train.cpu(), train_preds.cpu())
            test_accuracy = accuracy_score(y_test.cpu(), test_preds.cpu())
            train_acc.append(train_accuracy)
            test_acc.append(test_accuracy)
            # losses
            train_loss = loss.item()
            val_loss = criterion(model(X_test), y_test).item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Loss-based decision logic replacing accuracy-only heuristics
        if epoch > look_back and not lock:
            # moving-average window and helper
            ma_window = min(10, look_back)
            if len(train_losses) >= ma_window:
                # compute moving averages
                ma_train = np.convolve(train_losses, np.ones(ma_window)/ma_window, mode='valid')
                ma_val = np.convolve(val_losses, np.ones(ma_window)/ma_window, mode='valid')
                # compute recent slopes (difference of last two MA points)
                if len(ma_train) >= 2:
                    train_slope = ma_train[-1] - ma_train[-2]
                    val_slope = ma_val[-1] - ma_val[-2]
                else:
                    train_slope = 0.0
                    val_slope = 0.0

                # gap = val - train (positive if val loss higher)
                gap = ma_val[-1] - ma_train[-1]

                # plateau detection: both losses stop decreasing significantly
                if abs(train_slope) < 1e-4 and abs(val_slope) < 1e-4:
                    has_plateaued = True
                    num_plateued += 1
                    if current_class_index > goal_classes:
                        final_dataset_reached = True
                        final_plateau_count += 1
                        print("All classes have been added. Continuing training on final dataset.")

                # UNDERFITTING (loss-based): both losses are high and decreasing slowly
                # or both plateau but still high absolute loss
                if (train_slope > -0.001 and val_slope > -0.001) and (ma_train[-1] > 0.4) and num_plateued >= 3:
                    has_underfit = True
                    num_plateued = 0
                    final_plateau_count = 0
                    print(f"[train] Underfitting detected at epoch {epoch}. train_loss={ma_train[-1]:.4f}, val_loss={ma_val[-1]:.4f}, gap={gap:.4f}")
                    changed = model.add_hidden_neurons()
                    if changed:
                        optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])
                        print(f"[train] Model structure changed: added neurons. Hidden layers: {len(model.hidden_layers)}")
                        lock = True
                        lock_epochs = 200

                # OVERFITTING (loss-based): train loss keeps decreasing but val loss increases (gap grows)
                if (train_slope < -0.001 and val_slope > 0.001 and gap > 0.2) or (ma_train[-1] < 0.05 and gap > 0.2) or num_plateued > 20 or train_accuracy > 0.99:
                    has_overfit = True
                    num_plateued = 0
                    final_plateau_count = 0
                    print(f"[train] Overfitting detected at epoch {epoch}. train_loss={ma_train[-1]:.4f}, val_loss={ma_val[-1]:.4f}, gap={gap:.4f}")
                    if current_class_index <= goal_classes:
                        print(f"Adding new class {current_class_index} to the dataset.")
                        combine_new_class_to_dataset(all_train_dfs[current_class_index], all_test_dfs[current_class_index])
                        model.update_output_layer(len(torch.unique(y_train)))
                        optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])
                        current_class_index += 1
                        if current_class_index > goal_classes:
                            final_dataset_reached = True
                            print("Reached final combined dataset.")
                        lock_epochs = 50
                    else:
                        print("All classes have been added. Cannot add more classes. Continuing training.")
                        final_dataset_reached = True
                        lock = True
                        lock_epochs = 20

            # if any of the flags are set, lock further checks
            if has_underfit or has_overfit or has_plateaued:
                lock = True
                if has_underfit:
                    lock_epochs = max(lock_epochs, 200)
                else:
                    lock_epochs = max(lock_epochs, 10)
        else:
            if lock:
                lock_epochs -= 1
                if lock_epochs <= 0:
                    lock = False

        # reset flags for next epoch
        has_underfit = False
        has_overfit = False
        has_plateaued = False

        # optional stopping: if final dataset reached and we've seen multiple plateau detections, stop
        if final_dataset_reached and final_plateau_count >= 30:
            print("Final dataset plateaued multiple times — stopping training.")
            break

    print("Training finished. Reached class index:", current_class_index)
    return train_acc, test_acc, train_losses, val_losses
