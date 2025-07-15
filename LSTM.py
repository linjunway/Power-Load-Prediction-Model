import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import copy
from calendar import monthrange, month_name
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 5
BATCH_SIZE = 16
NUM_EPOCHS = 150  # Keep NUM_EPOCHS reasonable for demonstration
LEARNING_RATE = 0.001
LSTM_HIDDEN_SIZE = 128
ATTENTION_SIZE = 64
NUM_LSTM_LAYERS = 2
BIDIRECTIONAL_LSTM = True
DROPOUT_RATE = 0.25

NUM_POWER_FEATURES = 96
USE_YEARLY_LAG = True
NUM_LAG_FEATURES = NUM_POWER_FEATURES if USE_YEARLY_LAG else 0
NUM_TOTAL_FEATURES_MODEL_INPUT = NUM_POWER_FEATURES + NUM_LAG_FEATURES

# For the new validation plot:
NUM_VALIDATION_EVEN_DAYS_TO_PLOT = 3  # How many individual even days from validation to plot in detail

MONTHS_TO_PREDICT_LIST = [
    (2024, 6), (2024, 7), (2024, 8), (2024, 9), (2024, 10), (2024, 11)
]


# --- 1. Load Data and Preprocessing ---
def load_and_preprocess_data(file_path, sheet_name=0, is_train_or_val=True, is_odd_days_only_check_file=False):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}");
        return None
    except ValueError as e:
        print(f"Error loading sheet '{sheet_name}' from {file_path}: {e}");
        return None
    if 'Date' not in df.columns: print(f"Error: 'Date' column missing in {file_path}"); return None
    df['Date'] = pd.to_datetime(df['Date'].astype(str).str.split(' ').str[0], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values(by='Date').reset_index(drop=True)
    power_cols_list = [f'Power{i}' for i in range(1, NUM_POWER_FEATURES + 1)]
    if not is_odd_days_only_check_file:  # For train, val, or full check files
        for col in power_cols_list:
            if col not in df.columns: df[col] = np.nan  # Ensure all power columns exist
    if is_train_or_val:
        # Ensure all power columns are present for train/val before processing
        for col in power_cols_list:
            if col not in df.columns: df[col] = np.nan
        # Convert to numeric, drop rows if ANY power reading is NaN, clip negatives
        for col in power_cols_list:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=power_cols_list)  # Critical: drop rows with ANY NaN in power columns for train/val
        for col in power_cols_list:
            if col in df.columns: df[col] = df[col].apply(lambda x: max(0, x) if pd.notnull(x) else x)
    else:  # For check/operational data (potentially with only odd days or other NaNs)
        for col in power_cols_list:
            if col in df.columns:  # Only process columns that exist
                mask = df[col].notna()  # Create a mask for non-NaN values
                df.loc[mask, col] = pd.to_numeric(df.loc[mask, col], errors='coerce')  # Convert only non-NaN
    expected_cols = ['Date'] + power_cols_list
    if is_odd_days_only_check_file:  # If it's the special odd-days file, keep only existing power columns
        final_cols = ['Date'] + [col for col in power_cols_list if col in df.columns]
        df = df[final_cols]
    else:  # For other files, ensure all expected columns are present, adding NaNs if necessary
        for col in expected_cols:
            if col not in df.columns: df[col] = np.nan
        df = df[expected_cols]
    return df


# --- 2. Prepare Prediction Template ---
def prepare_prediction_template(target_month_year_str, odd_days_df_input):
    try:
        year, month = map(int, target_month_year_str.split('-'))
    except ValueError:
        print(f"Error: Invalid target_month_year_str: {target_month_year_str}.");
        return None
    num_days = monthrange(year, month)[1]
    all_dates = pd.to_datetime([f"{target_month_year_str}-{d:02d}" for d in range(1, num_days + 1)])
    p_cols = [f'Power{i}' for i in range(1, NUM_POWER_FEATURES + 1)]
    template_df = pd.DataFrame({'Date': all_dates})
    for col in p_cols: template_df[col] = np.nan
    if odd_days_df_input is not None and not odd_days_df_input.empty:
        odd_df = odd_days_df_input.copy();
        odd_df['Date'] = pd.to_datetime(odd_df['Date'])
        # Filter for the specific target month AND year
        odd_df_month = odd_df[(odd_df['Date'].dt.year == year) & (odd_df['Date'].dt.month == month)].copy()
        if not odd_df_month.empty:
            print(
                f"Found {len(odd_df_month)} entries from odd_days_input for {target_month_year_str} to merge into template.")
            template_idx = template_df.set_index('Date')
            cols_update = [col for col in odd_df_month.columns if
                           col in p_cols]  # Ensure only power columns are used for update
            odd_merge_idx = odd_df_month.set_index('Date')[cols_update]
            template_idx.update(odd_merge_idx, overwrite=True)
            template_df = template_idx.reset_index()
        else:
            print(f"No entries for {target_month_year_str} in odd_days_input for merging.")
    else:
        print(f"Warning: odd_days_df_input empty/None for {target_month_year_str}, template will be all NaNs.")
    return template_df.sort_values(by='Date').reset_index(drop=True)


# --- 3. Create Training/Validation Sequences ---
def create_sequences_with_lag_adapted(data_df, power_cols_list, seq_length, use_yearly_lag_flag):
    xs, ys = [], []
    num_power_features_dynamic = len(power_cols_list)
    if not isinstance(data_df.index, pd.DatetimeIndex):
        print("Error in create_sequences: data_df must have a DatetimeIndex.")
        return np.array(xs), np.array(ys)
    possible_target_dates = data_df.index[seq_length:]
    for target_date in possible_target_dates:
        end_of_x_sequence_date = target_date - pd.Timedelta(days=1)
        start_of_x_sequence_date = end_of_x_sequence_date - pd.Timedelta(days=seq_length - 1)
        required_x_dates = pd.date_range(start=start_of_x_sequence_date, end=end_of_x_sequence_date)
        if not all(d in data_df.index for d in required_x_dates) or target_date not in data_df.index:
            continue
        sequence_data_power_df = data_df.loc[required_x_dates, power_cols_list]
        if sequence_data_power_df.isnull().values.any() or len(sequence_data_power_df) != seq_length:
            continue
        x_current_power_features = sequence_data_power_df.values
        if use_yearly_lag_flag:
            sequence_data_lag_list = []
            for j in range(seq_length):
                date_in_seq = start_of_x_sequence_date + pd.Timedelta(days=j)
                lag_date = date_in_seq - pd.DateOffset(years=1)
                try:
                    lag_data_day = data_df.loc[lag_date, power_cols_list].values
                    if np.isnan(lag_data_day).any():
                        sequence_data_lag_list.append(np.zeros(num_power_features_dynamic))
                    else:
                        sequence_data_lag_list.append(lag_data_day)
                except KeyError:
                    sequence_data_lag_list.append(np.zeros(num_power_features_dynamic))
            sequence_data_lag_np = np.array(sequence_data_lag_list)
            if sequence_data_lag_np.shape[0] != seq_length: continue
            x_combined_features = np.concatenate((x_current_power_features, sequence_data_lag_np), axis=1)
        else:
            x_combined_features = x_current_power_features
        y_target_series = data_df.loc[target_date, power_cols_list]
        if y_target_series.isnull().values.any():
            continue
        y_target = y_target_series.values
        xs.append(x_combined_features);
        ys.append(y_target)
    return np.array(xs), np.array(ys)


# --- 4. PyTorch Dataset and Model ---
class PowerDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class Attention(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_network = nn.Sequential(nn.Linear(feature_dim, attention_dim), nn.Tanh(),
                                               nn.Linear(attention_dim, 1))

    def forward(self, lstm_output):
        attention_scores = self.attention_network(lstm_output)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, attention_size, num_layers, output_size_model, dropout_rate,
                 bidirectional):
        super(AdvancedLSTMModel, self).__init__()
        self.lstm_hidden_size, self.num_layers, self.bidirectional = lstm_hidden_size, num_layers, bidirectional
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0, bidirectional=bidirectional)
        lstm_output_dim = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.attention = Attention(lstm_output_dim, attention_size)
        self.fc = nn.Linear(lstm_output_dim, output_size_model)

    def forward(self, x):
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.lstm_hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0));
        context_vector, _ = self.attention(lstm_out);
        out = self.fc(context_vector)
        return out


# --- 5. Prediction and Filling Function (for operational use) ---
def predict_and_fill_with_lag(target_month_year_str, df_operational_scaled_with_history, trained_model, data_scaler,
                              seq_len, power_cols_list, use_yearly_lag_flag, dev):
    trained_model.eval()
    df_for_iterative_prediction_scaled = df_operational_scaled_with_history.copy()
    target_start_dt = pd.to_datetime(target_month_year_str + "-01")
    num_days_in_target_month = monthrange(target_start_dt.year, target_start_dt.month)[1]
    target_end_dt = pd.to_datetime(f"{target_month_year_str}-{num_days_in_target_month:02d}")
    print(f"Iteratively predicting for: {target_start_dt.strftime('%Y-%m-%d')} to {target_end_dt.strftime('%Y-%m-%d')}")
    model_outputs_scaled_for_target_month = []
    dates_predicted_in_target_month = []
    for current_pred_date in pd.date_range(start=target_start_dt, end=target_end_dt):
        if current_pred_date in df_for_iterative_prediction_scaled.index and \
                pd.notna(
                    df_for_iterative_prediction_scaled.loc[current_pred_date, power_cols_list[0]]):  # Already has data
            # We still want to record its date if it's in the target month, but not predict it.
            # The output processing logic will handle merging later. Here we just skip prediction.
            continue  # Skip if data already exists (e.g. odd day given)

        x_sequence_end_date = current_pred_date - pd.Timedelta(days=1)
        x_sequence_start_date = x_sequence_end_date - pd.Timedelta(days=seq_len - 1)
        required_x_hist_dates = pd.date_range(start=x_sequence_start_date, end=x_sequence_end_date)

        if not all(d in df_for_iterative_prediction_scaled.index for d in required_x_hist_dates):
            print(
                f"    Warning: Insufficient history for {current_pred_date.strftime('%Y-%m-%d')}. Cannot predict. Filling with NaN.")
            model_outputs_scaled_for_target_month.append(np.full(len(power_cols_list), np.nan))
            dates_predicted_in_target_month.append(current_pred_date)
            # Ensure this date exists in the df for iterative filling, marked as NaN
            if current_pred_date not in df_for_iterative_prediction_scaled.index:
                nan_row = pd.DataFrame(np.full((1, len(power_cols_list)), np.nan), columns=power_cols_list,
                                       index=[current_pred_date])
                df_for_iterative_prediction_scaled = pd.concat(
                    [df_for_iterative_prediction_scaled, nan_row]).sort_index()
            else:  # If it exists but was perhaps an all-NaN row initially
                df_for_iterative_prediction_scaled.loc[current_pred_date, power_cols_list] = np.nan
            continue

        current_x_power_scaled = df_for_iterative_prediction_scaled.loc[required_x_hist_dates, power_cols_list].values
        if np.isnan(current_x_power_scaled).any():
            print(
                f"    Warning: NaN in power input sequence for {current_pred_date.strftime('%Y-%m-%d')}. Cannot predict. Filling with NaN.")
            model_outputs_scaled_for_target_month.append(np.full(len(power_cols_list), np.nan))
            dates_predicted_in_target_month.append(current_pred_date)
            if current_pred_date not in df_for_iterative_prediction_scaled.index:  # Should exist from template generally
                df_for_iterative_prediction_scaled.loc[current_pred_date, power_cols_list] = np.nan  # Mark as NaN
            else:
                df_for_iterative_prediction_scaled.loc[current_pred_date, power_cols_list] = np.nan
            continue

        if use_yearly_lag_flag:
            current_x_lag_scaled_list = []
            for j in range(seq_len):
                date_in_seq = x_sequence_start_date + pd.Timedelta(days=j)
                lag_date = date_in_seq - pd.DateOffset(years=1)
                try:
                    lag_values = df_for_iterative_prediction_scaled.loc[lag_date, power_cols_list].values
                    current_x_lag_scaled_list.append(np.nan_to_num(lag_values, nan=0.0))  # Fill NaN lag with 0
                except KeyError:
                    current_x_lag_scaled_list.append(np.zeros(len(power_cols_list)))
            current_x_lag_scaled = np.array(current_x_lag_scaled_list)
            current_x_combined_scaled = np.concatenate((current_x_power_scaled, current_x_lag_scaled), axis=1)
        else:
            current_x_combined_scaled = current_x_power_scaled

        if current_x_combined_scaled.shape[0] != seq_len or current_x_combined_scaled.ndim != 2 or \
                current_x_combined_scaled.shape[1] != NUM_TOTAL_FEATURES_MODEL_INPUT:
            print(
                f"    Error: Sequence for {current_pred_date.strftime('%Y-%m-%d')} wrong shape {current_x_combined_scaled.shape}. Expected ({seq_len}, {NUM_TOTAL_FEATURES_MODEL_INPUT}). Skipping.")
            model_outputs_scaled_for_target_month.append(np.full(len(power_cols_list), np.nan));
            dates_predicted_in_target_month.append(current_pred_date)
            if current_pred_date not in df_for_iterative_prediction_scaled.index:
                df_for_iterative_prediction_scaled.loc[current_pred_date, power_cols_list] = np.nan
            else:
                df_for_iterative_prediction_scaled.loc[current_pred_date, power_cols_list] = np.nan
            continue

        input_tensor = torch.tensor(current_x_combined_scaled, dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            predicted_y_scaled_tensor = trained_model(input_tensor)
        predicted_y_scaled_np = predicted_y_scaled_tensor.cpu().numpy().flatten()

        model_outputs_scaled_for_target_month.append(predicted_y_scaled_np);
        dates_predicted_in_target_month.append(current_pred_date)
        # Update df_for_iterative_prediction_scaled with this new prediction for subsequent days in the same month
        if current_pred_date not in df_for_iterative_prediction_scaled.index:
            new_row_df = pd.DataFrame([predicted_y_scaled_np], columns=power_cols_list, index=[current_pred_date])
            df_for_iterative_prediction_scaled = pd.concat(
                [df_for_iterative_prediction_scaled, new_row_df]).sort_index()
        else:
            df_for_iterative_prediction_scaled.loc[current_pred_date, power_cols_list] = predicted_y_scaled_np

    model_outputs_scaled_np_array = np.array(model_outputs_scaled_for_target_month)

    if not dates_predicted_in_target_month:  # No dates were even attempted for prediction
        print(
            f"No dates were marked for prediction in {target_month_year_str}. Returning empty DF for model predictions.")
        return pd.DataFrame(columns=['Date'] + power_cols_list).set_index('Date').reset_index()

    # Filter out rows that are all NaN from model_outputs_scaled_np_array before inverse transform
    valid_predictions_mask = ~np.isnan(model_outputs_scaled_np_array).all(axis=1)
    if not np.any(valid_predictions_mask):
        print(
            f"All attempted predictions for {target_month_year_str} resulted in NaN. Returning empty DF for model predictions.")
        return pd.DataFrame(columns=['Date'] + power_cols_list).set_index('Date').reset_index()

    model_outputs_to_transform = model_outputs_scaled_np_array[valid_predictions_mask]
    dates_for_valid_predictions = pd.DatetimeIndex(np.array(dates_predicted_in_target_month)[valid_predictions_mask])

    if model_outputs_to_transform.size == 0:  # Should be caught by np.any(valid_predictions_mask) but as a safeguard
        print(f"No valid values to transform for {target_month_year_str}. Returning empty DF for model predictions.")
        return pd.DataFrame(columns=['Date'] + power_cols_list).set_index('Date').reset_index()

    model_predictions_original_scale = data_scaler.inverse_transform(model_outputs_to_transform)
    model_predictions_original_scale[model_predictions_original_scale < 0] = 0  # Clip negatives

    df_model_predictions_month = pd.DataFrame(data=model_predictions_original_scale, columns=power_cols_list,
                                              index=dates_for_valid_predictions)
    df_model_predictions_month.index.name = "Date"  # Set index name for reset_index()
    return df_model_predictions_month.reset_index()


# --- 6. Plot Continuous 15-minute Power for the Entire Month ---
def plot_continuous_monthly_power(month_df_cont, target_month_str_cont, p_cols_cont,
                                  check_df_odd_days_for_coloring):
    if month_df_cont.empty: print(f"Cannot plot continuous for {target_month_str_cont}: DataFrame empty."); return
    month_df_cont_plot = month_df_cont.dropna(subset=p_cols_cont, how='all').copy()
    if month_df_cont_plot.empty: print(
        f"No data to plot for {target_month_str_cont} after dropping all-NaN rows."); return
    odd_day_dates_with_data = set()
    if check_df_odd_days_for_coloring is not None and 'Date' in check_df_odd_days_for_coloring.columns:
        target_dt_for_filter = pd.to_datetime(target_month_str_cont + "-01")
        month_check_df = check_df_odd_days_for_coloring[
            (check_df_odd_days_for_coloring['Date'].dt.month == target_dt_for_filter.month) &
            (check_df_odd_days_for_coloring['Date'].dt.year == target_dt_for_filter.year)
            ]
        # For a day to be "given", it must have non-NaN data in check_df for at least one power column
        for idx_check, row_check in month_check_df.iterrows():
            if row_check[p_cols_cont].notna().any():
                odd_day_dates_with_data.add(row_check['Date'])

    plt.figure(figsize=(25, 8))
    plotted_given_label = False
    plotted_predicted_label = False
    month_df_cont_plot.sort_values(by='Date', inplace=True)

    for day_index, row in month_df_cont_plot.iterrows():
        day_dt = row['Date']
        day_power_readings = row[p_cols_cont].values.astype(float)  # Ensure float for plotting NaNs correctly if any
        if np.isnan(day_power_readings).all():
            continue  # Skip days with no power data at all

        is_given_day = day_dt in odd_day_dates_with_data
        current_day_color = 'gold' if is_given_day else 'dodgerblue'
        current_day_label_base = 'Given Data (Odd Days)' if is_given_day else 'Predicted Data (Even Days / Filled)'

        timestamps_for_day = pd.to_datetime([day_dt + pd.Timedelta(minutes=i * 15) for i in range(NUM_POWER_FEATURES)])

        current_label = None
        if is_given_day:
            if not plotted_given_label:
                current_label = current_day_label_base
                plotted_given_label = True
        else:  # Predicted day
            if not plotted_predicted_label:
                current_label = current_day_label_base
                plotted_predicted_label = True

        plt.plot(timestamps_for_day, day_power_readings, color=current_day_color, linestyle='-', linewidth=1.2,
                 label=current_label)

    target_datetime_title = pd.to_datetime(target_month_str_cont)
    plt.title(
        f'Continuous 15-Min Power: {month_name[target_datetime_title.month]} {target_datetime_title.year} (Given vs. Predicted)');
    plt.xlabel('Timestamp');
    plt.ylabel('Power Value');
    plt.xticks(rotation=30, ha="right");
    if plotted_given_label or plotted_predicted_label: plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5);
    plt.tight_layout();
    plt.show()


# --- Main Script Execution ---
train_df_raw = load_and_preprocess_data('preprocessed_data_2.xlsx', sheet_name='Sheet1', is_train_or_val=True)
val_df_raw = load_and_preprocess_data('validation_data.xlsx', sheet_name='Validation', is_train_or_val=True)
check_df_odd_days_only_master = load_and_preprocess_data('check_data_to_students.xlsx', sheet_name='Check2',
                                                         is_train_or_val=False, is_odd_days_only_check_file=True)
if train_df_raw is None or train_df_raw.empty: print("Critical Error: Training data missing. Exiting."); exit()
power_cols = [f'Power{i}' for i in range(1, NUM_POWER_FEATURES + 1)]
dfs_to_concat = [train_df_raw]
if val_df_raw is not None and not val_df_raw.empty:
    dfs_to_concat.append(val_df_raw)
else:
    print("Warning: Validation data not loaded or empty. `df_historical_combined` will only contain training data.")
df_historical_combined = pd.concat(dfs_to_concat, ignore_index=True).sort_values(by='Date').reset_index(drop=True)
df_historical_combined.set_index('Date', inplace=True)
df_historical_combined = df_historical_combined[~df_historical_combined.index.duplicated(keep='first')]
first_prediction_year, first_prediction_month = MONTHS_TO_PREDICT_LIST[0]
training_cutoff_date_main = pd.to_datetime(f"{first_prediction_year}-{first_prediction_month:02d}-01") - pd.Timedelta(
    days=1)
df_for_scaler_and_training_seq = df_historical_combined[
    df_historical_combined.index <= training_cutoff_date_main].copy()
if df_for_scaler_and_training_seq.empty: print(
    f"Error: No data for scaler fitting and training sequences before {training_cutoff_date_main}. Check data and MONTHS_TO_PREDICT_LIST."); exit()
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_for_scaler_and_training_seq[power_cols].values)  # Fit scaler only on training portion
# Transform all historical data (train + val) using the fitted scaler
df_historical_combined.loc[:, power_cols] = scaler.transform(df_historical_combined.loc[:, power_cols].values)
print("Historical data (train and val, if val loaded) scaled and df_historical_combined updated.")
X_train, y_train = create_sequences_with_lag_adapted(
    df_historical_combined[df_historical_combined.index <= training_cutoff_date_main], power_cols, SEQUENCE_LENGTH,
    USE_YEARLY_LAG)
if X_train.shape[0] == 0: print(
    "Error: No training sequences created. Check data, sequence length, and cutoff date. Exiting."); exit()
print(f"Created training sequences: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
train_loader = DataLoader(
    PowerDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
model = AdvancedLSTMModel(NUM_TOTAL_FEATURES_MODEL_INPUT, LSTM_HIDDEN_SIZE, ATTENTION_SIZE, NUM_LSTM_LAYERS,
                          NUM_POWER_FEATURES, DROPOUT_RATE, BIDIRECTIONAL_LSTM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-7)
print(f"--- Starting Training on {DEVICE} (Advanced Model) ---")
train_losses_plot = []
for epoch in range(NUM_EPOCHS):
    model.train();
    epoch_loss = 0
    for batch_X_train, batch_y_train in train_loader:
        batch_X, batch_y = batch_X_train.to(DEVICE), batch_y_train.to(DEVICE)
        optimizer.zero_grad();
        outputs = model(batch_X);
        loss = criterion(outputs, batch_y)
        loss.backward();
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0);
        optimizer.step()
        epoch_loss += loss.item()
    avg_epoch_loss = epoch_loss / len(train_loader);
    train_losses_plot.append(avg_epoch_loss);
    scheduler.step(avg_epoch_loss)
    if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1: print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.7f}")
print("Training complete.")
plt.figure(figsize=(10, 5));
plt.plot(train_losses_plot, label='Train Loss');
plt.title('Model Training Loss');
plt.xlabel('Epochs');
plt.ylabel('Loss (MSE)');
plt.legend();
plt.grid(True);
plt.savefig("training_loss_plot.png");
print("Training loss plot saved to training_loss_plot.png");
plt.show(block=False)  # block=False for potentially faster run

if val_df_raw is not None and not val_df_raw.empty:
    print("\n--- Predicting and Plotting for Validation Set Even Days ---")
    val_df_raw_unscaled_for_plot = load_and_preprocess_data('validation_data.xlsx', sheet_name='Validation',
                                                            is_train_or_val=True)  # Reload for unscaled actuals
    if val_df_raw_unscaled_for_plot is not None and not val_df_raw_unscaled_for_plot.empty:
        val_df_raw_unscaled_for_plot.set_index('Date', inplace=True);
        val_df_raw_unscaled_for_plot = val_df_raw_unscaled_for_plot[
            ~val_df_raw_unscaled_for_plot.index.duplicated(keep='first')]

        scaled_val_predictions_list = [];
        val_prediction_dates = []
        model.eval()
        with torch.no_grad():
            # Predict only for even days present in the validation set
            even_val_dates = [d for d in val_df_raw_unscaled_for_plot.index if d.day % 2 == 0]
            for val_date_to_predict in even_val_dates:
                x_sequence_end_date = val_date_to_predict - pd.Timedelta(days=1)
                x_sequence_start_date = x_sequence_end_date - pd.Timedelta(days=SEQUENCE_LENGTH - 1)
                required_x_hist_dates = pd.date_range(start=x_sequence_start_date, end=x_sequence_end_date)

                if not all(d in df_historical_combined.index for d in required_x_hist_dates):
                    scaled_val_predictions_list.append(np.full(NUM_POWER_FEATURES, np.nan))
                    val_prediction_dates.append(val_date_to_predict)
                    continue

                current_x_power_scaled = df_historical_combined.loc[required_x_hist_dates, power_cols].values
                if np.isnan(current_x_power_scaled).any():
                    scaled_val_predictions_list.append(np.full(NUM_POWER_FEATURES, np.nan))
                    val_prediction_dates.append(val_date_to_predict)
                    continue

                if USE_YEARLY_LAG:
                    current_x_lag_scaled_list = []
                    for j_lag in range(SEQUENCE_LENGTH):
                        date_in_seq_lag = x_sequence_start_date + pd.Timedelta(days=j_lag)
                        lag_date_val = date_in_seq_lag - pd.DateOffset(years=1)
                        try:
                            lag_values_val = df_historical_combined.loc[lag_date_val, power_cols].values
                            current_x_lag_scaled_list.append(np.nan_to_num(lag_values_val, nan=0.0))
                        except KeyError:
                            current_x_lag_scaled_list.append(np.zeros(NUM_POWER_FEATURES))
                    current_x_lag_scaled_val = np.array(current_x_lag_scaled_list)
                    current_x_combined_scaled_val = np.concatenate((current_x_power_scaled, current_x_lag_scaled_val),
                                                                   axis=1)
                else:
                    current_x_combined_scaled_val = current_x_power_scaled

                if current_x_combined_scaled_val.ndim == 1: current_x_combined_scaled_val = current_x_combined_scaled_val.reshape(
                    1, -1)  # Should be (seq_len, features)
                if current_x_combined_scaled_val.shape[0] != SEQUENCE_LENGTH or current_x_combined_scaled_val.shape[
                    1] != NUM_TOTAL_FEATURES_MODEL_INPUT:
                    scaled_val_predictions_list.append(np.full(NUM_POWER_FEATURES, np.nan))
                    val_prediction_dates.append(val_date_to_predict)
                    continue

                input_tensor_val = torch.tensor(current_x_combined_scaled_val, dtype=torch.float32).unsqueeze(0).to(
                    DEVICE)
                predicted_y_scaled_tensor_val = model(input_tensor_val)
                predicted_y_scaled_np_val = predicted_y_scaled_tensor_val.cpu().numpy().flatten()
                scaled_val_predictions_list.append(predicted_y_scaled_np_val)
                val_prediction_dates.append(val_date_to_predict)

        if scaled_val_predictions_list:
            df_val_even_day_preds_scaled = pd.DataFrame(scaled_val_predictions_list,
                                                        index=pd.DatetimeIndex(val_prediction_dates),
                                                        columns=power_cols)
            df_val_even_day_preds_scaled.dropna(how='all', inplace=True)  # Remove rows if all predictions were NaN

            if not df_val_even_day_preds_scaled.empty:
                val_preds_unscaled_values = scaler.inverse_transform(df_val_even_day_preds_scaled.values)
                val_preds_unscaled_values[val_preds_unscaled_values < 0] = 0  # Clip negatives
                df_val_even_day_preds_unscaled = pd.DataFrame(val_preds_unscaled_values,
                                                              index=df_val_even_day_preds_scaled.index,
                                                              columns=power_cols)

                actual_even_day_val_unscaled = val_df_raw_unscaled_for_plot[
                    val_df_raw_unscaled_for_plot.index.day % 2 == 0].copy()  # Ensure it's a copy

                common_indices = actual_even_day_val_unscaled.index.intersection(df_val_even_day_preds_unscaled.index)

                if not common_indices.empty:
                    actuals_for_metric_raw = actual_even_day_val_unscaled.loc[
                        common_indices, power_cols].values.flatten()
                    preds_for_metric_raw = df_val_even_day_preds_unscaled.loc[
                        common_indices, power_cols].values.flatten()

                    # Filter out any NaNs that might have occurred if a day had partial NaNs in actuals or preds
                    valid_metric_mask = ~np.isnan(actuals_for_metric_raw) & ~np.isnan(preds_for_metric_raw)
                    actuals_for_metric = actuals_for_metric_raw[valid_metric_mask]
                    preds_for_metric = preds_for_metric_raw[valid_metric_mask]

                    if actuals_for_metric.size > 0 and preds_for_metric.size > 0:
                        val_mse = mean_squared_error(actuals_for_metric, preds_for_metric)
                        val_mae = mean_absolute_error(actuals_for_metric, preds_for_metric)
                        print(f"Validation Even Day Prediction MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")

                        # --- BEGIN: Scatter plot of Actual vs Predicted for Validation Even Days ---
                        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
                        ax_scatter.scatter(actuals_for_metric, preds_for_metric, alpha=0.5, label='Predictions',
                                           s=15)  # s for marker size

                        combined_values_scatter = np.concatenate((actuals_for_metric, preds_for_metric))
                        abs_min_scatter = np.min(combined_values_scatter)
                        abs_max_scatter = np.max(combined_values_scatter)

                        range_val_scatter = abs_max_scatter - abs_min_scatter
                        if range_val_scatter == 0:
                            padding_scatter = 0.5
                        else:
                            padding_scatter = range_val_scatter * 0.05

                        plot_min_lim_scatter = abs_min_scatter - padding_scatter
                        plot_max_lim_scatter = abs_max_scatter + padding_scatter

                        # Handle case where all data points are zero or very close
                        if abs_min_scatter == 0 and abs_max_scatter == 0:
                            plot_min_lim_scatter = -0.5
                            plot_max_lim_scatter = 0.5
                        # Ensure some visual space if min and max are too close after padding
                        elif plot_max_lim_scatter - plot_min_lim_scatter < 0.1:  # Small arbitrary threshold
                            plot_min_lim_scatter -= 0.05
                            plot_max_lim_scatter += 0.05

                        ax_scatter.plot([plot_min_lim_scatter, plot_max_lim_scatter],
                                        [plot_min_lim_scatter, plot_max_lim_scatter], 'k--', lw=2, label='Ideal (y=x)')
                        ax_scatter.set_xlabel('Actual Power')
                        ax_scatter.set_ylabel('Predicted Power')
                        ax_scatter.set_title('Validation Set: Actual vs. Predicted Power (Even Days)')

                        ax_scatter.set_xlim(plot_min_lim_scatter, plot_max_lim_scatter)
                        ax_scatter.set_ylim(plot_min_lim_scatter, plot_max_lim_scatter)

                        # Add secondary y-axis like the example image
                        ax_scatter2 = ax_scatter.twinx()
                        ax_scatter2.set_ylim(ax_scatter.get_ylim())  # Match limits of primary y-axis
                        ax_scatter2.set_ylabel("Power")  # Label for the right axis

                        ax_scatter.legend(loc='best')  # Let matplotlib decide the best location for legend
                        ax_scatter.grid(True)

                        plt.tight_layout()  # Adjust plot to ensure everything fits without overlapping
                        plt.savefig("validation_actual_vs_predicted_scatter.png")
                        print(
                            "Validation actual vs predicted scatter plot saved to validation_actual_vs_predicted_scatter.png")
                        plt.show(block=False)
                        # --- END: Scatter plot ---
                    else:
                        print(
                            "No valid (non-NaN) data points for actual vs predicted scatter plot after filtering common indices.")

                    num_days_to_plot_val_actual = min(len(common_indices), NUM_VALIDATION_EVEN_DAYS_TO_PLOT)
                    if num_days_to_plot_val_actual > 0:
                        selected_val_days_for_plot = common_indices[:num_days_to_plot_val_actual]
                        fig, axes = plt.subplots(num_days_to_plot_val_actual, 1,
                                                 figsize=(15, 5 * num_days_to_plot_val_actual), squeeze=False)
                        fig.suptitle(
                            'Validation: Actual (non-zero segments) vs. Predicted 15-Minute Profiles for Even Days',
                            fontsize=16)
                        time_intervals_x_axis_daily = np.arange(NUM_POWER_FEATURES)
                        for i_plot, day_to_plot in enumerate(selected_val_days_for_plot):
                            ax = axes[i_plot, 0]
                            actual_values_for_day = actual_even_day_val_unscaled.loc[day_to_plot, power_cols].values
                            predicted_values_for_day = df_val_even_day_preds_unscaled.loc[
                                day_to_plot, power_cols].values
                            ax.plot(time_intervals_x_axis_daily, predicted_values_for_day, label='Predicted Profile',
                                    color='red', linestyle='--')
                            start_segment_idx = -1;
                            plotted_actual_legend_label = False;
                            segment_label = None
                            for k_interval in range(NUM_POWER_FEATURES):
                                if actual_values_for_day[k_interval] > 1e-6 and start_segment_idx == -1:
                                    start_segment_idx = k_interval
                                elif actual_values_for_day[k_interval] <= 1e-6 and start_segment_idx != -1:
                                    segment_label = 'Actual Profile (non-zero)' if not plotted_actual_legend_label else ""
                                    ax.plot(time_intervals_x_axis_daily[start_segment_idx:k_interval],
                                            actual_values_for_day[start_segment_idx:k_interval],
                                            label=segment_label if segment_label else None, color='green',
                                            linestyle='-');
                                    if segment_label: plotted_actual_legend_label = True
                                    start_segment_idx = -1;
                                    segment_label = None
                            if start_segment_idx != -1:
                                segment_label = 'Actual Profile (non-zero)' if not plotted_actual_legend_label else ""
                                ax.plot(time_intervals_x_axis_daily[start_segment_idx:],
                                        actual_values_for_day[start_segment_idx:],
                                        label=segment_label if segment_label else None, color='green', linestyle='-');
                                if segment_label: plotted_actual_legend_label = True
                            if not plotted_actual_legend_label and np.any(actual_values_for_day > 1e-6): ax.plot([], [],
                                                                                                                 label='Actual Profile (non-zero)',
                                                                                                                 color='green',
                                                                                                                 linestyle='-')  # Ensure legend if data exists but all segments were too short
                            ax.set_title(f'Day: {day_to_plot.strftime("%Y-%m-%d")}');
                            ax.set_xlabel('15-Minute Interval Index (0-95 representing 00:00 to 23:45)');
                            ax.set_ylabel('Power Value');
                            ax.legend();
                            ax.grid(True)
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
                        plt.savefig("validation_even_days_profile_comparison_plot.png");
                        print(
                            "Validation even days profile comparison plot saved to validation_even_days_profile_comparison_plot.png");
                        plt.show(block=False)
                    else:
                        print("No common even days with predictions to plot for validation profile comparison.")

                    print("\n--- Plotting Continuous February Even Days from Validation Data (if available) ---")
                    feb_common_indices = [idx for idx in common_indices if idx.month == 2]
                    if feb_common_indices:
                        feb_common_indices.sort()
                        actual_feb_even_days_df = actual_even_day_val_unscaled.loc[feb_common_indices, power_cols]
                        pred_feb_even_days_df = df_val_even_day_preds_unscaled.loc[feb_common_indices, power_cols]
                        all_actual_feb_even_vals = []
                        all_pred_feb_even_vals = []
                        for day_idx in feb_common_indices:  # Ensure order
                            all_actual_feb_even_vals.extend(actual_feb_even_days_df.loc[day_idx].tolist())
                            all_pred_feb_even_vals.extend(pred_feb_even_days_df.loc[day_idx].tolist())

                        if all_actual_feb_even_vals:  # Check if there is data to plot
                            continuous_x_axis_feb = np.arange(len(all_actual_feb_even_vals))
                            plt.figure(figsize=(20, 7))
                            plt.plot(continuous_x_axis_feb, all_pred_feb_even_vals,
                                     label='Predicted Feb Even Days Profile', color='red', linestyle='--')
                            start_segment_feb_idx = -1;
                            plotted_actual_feb_legend_label = False;
                            segment_label_feb = None
                            for k_feb_interval in range(len(all_actual_feb_even_vals)):
                                if all_actual_feb_even_vals[k_feb_interval] > 1e-6 and start_segment_feb_idx == -1:
                                    start_segment_feb_idx = k_feb_interval
                                elif all_actual_feb_even_vals[k_feb_interval] <= 1e-6 and start_segment_feb_idx != -1:
                                    segment_label_feb = 'Actual Feb Even Days (non-zero)' if not plotted_actual_feb_legend_label else ""
                                    plt.plot(continuous_x_axis_feb[start_segment_feb_idx:k_feb_interval],
                                             all_actual_feb_even_vals[start_segment_feb_idx:k_feb_interval],
                                             label=segment_label_feb if segment_label_feb else None, color='green',
                                             linestyle='-');
                                    if segment_label_feb: plotted_actual_feb_legend_label = True
                                    start_segment_feb_idx = -1;
                                    segment_label_feb = None
                            if start_segment_feb_idx != -1:  # Plot any remaining segment
                                segment_label_feb = 'Actual Feb Even Days (non-zero)' if not plotted_actual_feb_legend_label else ""
                                plt.plot(continuous_x_axis_feb[start_segment_feb_idx:],
                                         all_actual_feb_even_vals[start_segment_feb_idx:],
                                         label=segment_label_feb if segment_label_feb else None, color='green',
                                         linestyle='-');
                                if segment_label_feb: plotted_actual_feb_legend_label = True
                            if not plotted_actual_feb_legend_label and np.any(
                                np.array(all_actual_feb_even_vals) > 1e-6): plt.plot([], [],
                                                                                     label='Actual Feb Even Days (non-zero)',
                                                                                     color='green', linestyle='-')

                            feb_years = sorted(list(set(idx.year for idx in feb_common_indices)))
                            title_years = ", ".join(map(str, feb_years))
                            plt.title(
                                f'Validation: Continuous Actual vs. Predicted Profiles for February Even Days ({title_years})')
                            plt.xlabel(f'15-Minute Interval Index (across {len(feb_common_indices)} even days)')
                            plt.ylabel('Power Value');
                            plt.legend();
                            plt.grid(True);
                            plt.tight_layout()
                            plt.savefig("validation_february_even_days_continuous_plot.png")
                            print(
                                f"Validation February ({title_years}) even days continuous plot saved to validation_february_even_days_continuous_plot.png")
                            plt.show(block=False)
                        else:
                            print(
                                "No non-zero actual data found for February even days in the common indices to plot continuously.")
                    else:
                        print("No common even days found for February in the validation set.")
                else:
                    print(
                        "No common indices found between actual even days and predicted even days for validation metrics/plot.")
            else:
                print("No valid (non-NaN) even day predictions made for the validation set to plot/evaluate.")
        else:
            print("No even day predictions were attempted or successful for the validation set (list is empty).")
    else:
        print("Unscaled validation data for plotting not available (val_df_raw_unscaled_for_plot is None or empty).")
else:
    print("Validation data (val_df_raw) not loaded or empty. Skipping validation even day prediction and plotting.")

all_predictions_for_excel = pd.DataFrame()
df_operational_for_pred_scaled_master = df_historical_combined.copy()  # Start with all scaled historical data

for year_to_predict, month_to_predict in MONTHS_TO_PREDICT_LIST:
    CURRENT_TARGET_MONTH_STR = f"{year_to_predict}-{month_to_predict:02d}";
    print(f"\n\n===== OPERATIONAL PROCESSING TARGET MONTH: {month_name[month_to_predict]} {year_to_predict} =====")

    # 1. Prepare template for the current month, incorporating any odd day data from check_df
    complete_check_template_unscaled_current_month = prepare_prediction_template(
        CURRENT_TARGET_MONTH_STR,
        check_df_odd_days_only_master
    )
    if complete_check_template_unscaled_current_month is None or complete_check_template_unscaled_current_month.empty:
        print(f"Error: Failed to prepare template for {CURRENT_TARGET_MONTH_STR}. Skipping this month.")
        continue

    # 2. Scale the known (odd day) data from the template and update/add to the master operational df
    template_to_process_scaled_current = complete_check_template_unscaled_current_month.set_index('Date').copy()
    # Identify rows with actual data (from odd days in check_df)
    rows_with_odd_data_unscaled_current = template_to_process_scaled_current.dropna(subset=power_cols, how='all')

    if not rows_with_odd_data_unscaled_current.empty:
        # Scale these odd day values
        # Fill potential NaNs within a row (if some power features are NaN but not all) with 0 before scaling
        scaled_odd_day_values_current = scaler.transform(
            rows_with_odd_data_unscaled_current[power_cols].fillna(0).values
        )
        df_scaled_odd_days_current = pd.DataFrame(
            scaled_odd_day_values_current, columns=power_cols,
            index=rows_with_odd_data_unscaled_current.index
        )
        # Update the master operational DataFrame with these scaled odd day values
        # This ensures that df_operational_for_pred_scaled_master has the latest known (scaled) data
        df_operational_for_pred_scaled_master.update(df_scaled_odd_days_current)
        print(
            f"Updated master operational scaled data with {len(df_scaled_odd_days_current)} scaled odd day entries for {CURRENT_TARGET_MONTH_STR}.")

    # Ensure all dates from the current month's template are in the master df, adding NaNs for days to be predicted
    for date_idx_op_current in template_to_process_scaled_current.index:
        if date_idx_op_current not in df_operational_for_pred_scaled_master.index:
            # If date is not in master, add it with its data from template (could be NaNs or odd day data)
            # If it's odd day data, it would have been scaled and updated above. If NaNs, it's fine.
            # This step ensures the master df has rows for all days in the current target month.
            new_row_df = pd.DataFrame(template_to_process_scaled_current.loc[
                                          [date_idx_op_current], power_cols])  # Preserve existing values from template
            # Scale if it has data (e.g. odd day) that wasn't part of rows_with_odd_data_unscaled_current due to subset dropna
            if not new_row_df.isnull().all().all():  # if there's some data
                new_row_df[power_cols] = scaler.transform(new_row_df[power_cols].fillna(0).values)

            # If new_row_df was all NaNs, this keeps it as NaNs, which is correct for days to be predicted
            for col in power_cols:  # Ensure all power columns are there
                if col not in new_row_df.columns: new_row_df[col] = np.nan

            df_operational_for_pred_scaled_master = pd.concat([
                df_operational_for_pred_scaled_master,
                new_row_df[power_cols]  # Ensure only power_cols
            ])

    df_operational_for_pred_scaled_master.sort_index(inplace=True)
    df_operational_for_pred_scaled_master = df_operational_for_pred_scaled_master[
        ~df_operational_for_pred_scaled_master.index.duplicated(keep='first')
    ]

    # 3. Ensure history needed for prediction exists, possibly backfilling from check_df if missing in train/val
    first_day_of_current_target_month = pd.to_datetime(CURRENT_TARGET_MONTH_STR + "-01")
    for i_hist_check in range(1, SEQUENCE_LENGTH + 2):  # Check a bit more history
        day_needed = first_day_of_current_target_month - pd.Timedelta(days=i_hist_check)
        if day_needed not in df_operational_for_pred_scaled_master.index or \
                (day_needed in df_operational_for_pred_scaled_master.index and \
                 df_operational_for_pred_scaled_master.loc[day_needed, power_cols].isnull().all()):

            if check_df_odd_days_only_master is not None:
                prev_day_data_check_current = check_df_odd_days_only_master[
                    check_df_odd_days_only_master['Date'] == day_needed
                    ]
                if not prev_day_data_check_current.empty and not prev_day_data_check_current[
                    power_cols].isnull().all().all():
                    print(
                        f"Historical day {day_needed.strftime('%Y-%m-%d')} missing/NaN in master. Found in check_df. Scaling and adding.")
                    aligned_prev_day_unscaled = pd.DataFrame(index=[day_needed], columns=power_cols)
                    for p_col_hist in power_cols:  # Align columns
                        if p_col_hist in prev_day_data_check_current.columns:
                            aligned_prev_day_unscaled.loc[day_needed, p_col_hist] = prev_day_data_check_current.iloc[0][
                                p_col_hist]

                    scaled_prev_day_vals = scaler.transform(aligned_prev_day_unscaled.fillna(0).values)
                    df_prev_day_scaled_current = pd.DataFrame(scaled_prev_day_vals, columns=power_cols,
                                                              index=[day_needed])
                    df_operational_for_pred_scaled_master.update(df_prev_day_scaled_current)  # Update if index exists
                    if day_needed not in df_operational_for_pred_scaled_master.index:  # Add if new
                        df_operational_for_pred_scaled_master = pd.concat(
                            [df_operational_for_pred_scaled_master, df_prev_day_scaled_current])
                    df_operational_for_pred_scaled_master.sort_index(inplace=True)
                    df_operational_for_pred_scaled_master = df_operational_for_pred_scaled_master[
                        ~df_operational_for_pred_scaled_master.index.duplicated(keep='first')]

    # 4. Perform predictions for the current month
    df_model_predictions_current_month_unscaled = predict_and_fill_with_lag(
        CURRENT_TARGET_MONTH_STR,
        df_operational_for_pred_scaled_master,  # Use the updated master with scaled odd days and history
        model, scaler, SEQUENCE_LENGTH, power_cols,
        USE_YEARLY_LAG, DEVICE
    )

    # 5. Combine predictions with initial template (which has odd day data)
    final_current_month_output_df = complete_check_template_unscaled_current_month.copy()
    if not df_model_predictions_current_month_unscaled.empty:
        final_current_month_output_df.set_index('Date', inplace=True)
        df_model_predictions_current_month_unscaled.set_index('Date', inplace=True)
        # Update NaN values in final_df with predictions. Don't overwrite existing odd day data.
        final_current_month_output_df.update(df_model_predictions_current_month_unscaled, overwrite=False)
        final_current_month_output_df.reset_index(inplace=True)
    else:
        print(
            f"Warning: Model returned no NEW predictions for {CURRENT_TARGET_MONTH_STR}. Output relies on initial template content (odd days).")

    if final_current_month_output_df.empty:
        print(f"No data in final output for {CURRENT_TARGET_MONTH_STR}.")
    else:
        print(f"\n--- Final Data for {CURRENT_TARGET_MONTH_STR} ---");
        # print(final_current_month_output_df.head())
        nan_after_pred_curr = final_current_month_output_df[power_cols].isna().sum().sum();
        print(f"Total NaNs in power columns of final output for {CURRENT_TARGET_MONTH_STR}: {nan_after_pred_curr}");
        if nan_after_pred_curr > 0: print(
            "  Warning: Some NaNs remain in final output for this month's power data (could be due to insufficient history for early days).")

        all_predictions_for_excel = pd.concat([all_predictions_for_excel, final_current_month_output_df],
                                              ignore_index=True)

        print(f"\n--- Plotting Continuous 15-Minute Power Data for {CURRENT_TARGET_MONTH_STR} ---");
        plot_continuous_monthly_power(final_current_month_output_df, CURRENT_TARGET_MONTH_STR, power_cols,
                                      check_df_odd_days_only_master)

        # 6. Update master operational DF with scaled versions of the *filled* current month data for next iteration
        current_month_filled_unscaled_for_update = final_current_month_output_df.copy().set_index('Date')
        if not current_month_filled_unscaled_for_update.empty:
            # Fill any remaining NaNs with 0 before scaling, for consistency in master df
            power_data_to_scale = current_month_filled_unscaled_for_update[power_cols].fillna(0).values;
            scaled_power_data_for_master_update = scaler.transform(power_data_to_scale);
            df_current_month_filled_and_scaled_for_update = pd.DataFrame(
                scaled_power_data_for_master_update,
                columns=power_cols,
                index=current_month_filled_unscaled_for_update.index
            );
            df_operational_for_pred_scaled_master.update(df_current_month_filled_and_scaled_for_update);
            # Re-sort and de-duplicate just in case update introduced issues (shouldn't if indices are unique)
            df_operational_for_pred_scaled_master.sort_index(inplace=True)
            df_operational_for_pred_scaled_master = df_operational_for_pred_scaled_master[
                ~df_operational_for_pred_scaled_master.index.duplicated(keep='first')]
            print(
                f"Master operational scaled data (df_operational_for_pred_scaled_master) updated with scaled results from {month_name[month_to_predict]} {year_to_predict}.")

if not all_predictions_for_excel.empty:
    start_month_name_excel = month_name[MONTHS_TO_PREDICT_LIST[0][1]][:3];
    start_year_excel = MONTHS_TO_PREDICT_LIST[0][0];
    end_month_name_excel = month_name[MONTHS_TO_PREDICT_LIST[-1][1]][:3];
    end_year_excel = MONTHS_TO_PREDICT_LIST[-1][0];
    combined_filename = f"predictions_filled_{start_month_name_excel}{start_year_excel}_to_{end_month_name_excel}{end_year_excel}_advModel.xlsx";
    all_predictions_for_excel.sort_values(by='Date', inplace=True);
    all_predictions_for_excel.to_excel(combined_filename, index=False, engine='openpyxl');
    print(f"\nALL MONTHLY PREDICTIONS SAVED TO: '{combined_filename}'")
else:
    print("\nNo predictions were generated across any months to save to a combined file.")

# Ensure all plots are displayed if block=False was used
plt.show()