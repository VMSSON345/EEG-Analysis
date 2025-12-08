import mne
import pandas as pd
import numpy as np
import glob
import os
import re
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from mne.time_frequency import psd_array_welch

def extract_features_hmm(base_data_path: str, out_train: str, out_test: str):
    EXCLUDE_SUBJECTS = ['sub-02', 'sub-07', 'sub-08', 'sub-10', 'sub-16', 'sub-23', 'sub-25', 'sub-35']
    HMM_CHANNELS = ['Pz', 'Cz', 'Fz', 'P3', 'P4']
    N_STATES = 5
    TRAIN_RATIO = 0.8

    BANDS = {
        "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
    }
    ERP_WINDOW = (0.25, 0.6)

    epoch_files = glob.glob(os.path.join(base_data_path, '*_cleaned.fif'))
    if not epoch_files:
        print("Do not find any .fif files.")
        return
    print(f"Found {len(epoch_files)} files.")
    train_rows, test_rows = [], []

    for file_path in epoch_files:
        filename = os.path.basename(file_path)
        match = re.search(r'(sub[_-]?\d+|subject[_-]?\d+)', filename.lower())
        subject_id = match.group(0) if match else filename.split('_')[0]

        norm_id = subject_id.replace('_', '-').lower()
        if any(bad in norm_id for bad in EXCLUDE_SUBJECTS):
            print(f"[SKIP]: {subject_id}")
            continue

        epochs = mne.read_epochs(file_path, preload=True, verbose='ERROR')
        print(f"Processing: {subject_id}")
        try:
            epochs_subset = epochs['Standard', 'Target']
        except KeyError: continue

        target_ch_names = epochs_subset.ch_names
        sfreq = epochs.info['sfreq']
        n_trials = len(epochs_subset)

        # Nhãn
        event_id_map = epochs.event_id
        def get_label(code):
            if code == event_id_map.get('Standard'): return 0
            if code == event_id_map.get('Target'): return 1
            return -1
        labels = np.array([get_label(c) for c in epochs_subset.events[:, 2]])
        epochs_hmm = epochs_subset.copy().pick(HMM_CHANNELS)
        epochs_hmm.resample(100, verbose='ERROR')
        X_hmm_raw = epochs_hmm.get_data()
        n_trials_hmm, _, n_times_hmm = X_hmm_raw.shape

        scaler = StandardScaler()
        X_flat = X_hmm_raw.transpose(0, 2, 1).reshape(-1, len(HMM_CHANNELS))
        X_scaled_flat = scaler.fit_transform(X_flat)
        X_hmm_scaled = X_scaled_flat.reshape(n_trials_hmm, n_times_hmm, len(HMM_CHANNELS))

        split_idx = int(n_trials * TRAIN_RATIO)
        X_train_hmm = X_hmm_scaled[:split_idx]
        y_train_hmm = labels[:split_idx]

        def train_hmm_model(X_full, y_full, cls):
            X_cls = X_full[y_full == cls]
            if len(X_cls) < 5: return None
            dat = X_cls.reshape(-1, len(HMM_CHANNELS))
            lens = [n_times_hmm] * len(X_cls)
            model = hmm.GaussianHMM(n_components=N_STATES, covariance_type="diag", n_iter=10, random_state=42)
            try: model.fit(dat, lens); return model
            except: return None

        hmm_models = {
            0: train_hmm_model(X_train_hmm, y_train_hmm, 0),
            1: train_hmm_model(X_train_hmm, y_train_hmm, 1)
        }

        full_data = epochs_subset.get_data()
        t_start, t_end = epochs_subset.time_as_index(ERP_WINDOW)
        erp_full = full_data[:, :, t_start:t_end]
        erp_times = epochs_subset.times[t_start:t_end]

        psds_all, freqs = psd_array_welch(
            full_data, sfreq=sfreq, fmin=0.5, fmax=30.0, n_fft=256, verbose='ERROR'
        )

        for i in range(n_trials):
            row = {'subject_id': subject_id, 'trial_id': i, 'label': labels[i]}

            seq = X_hmm_scaled[i]
            for cls, model in hmm_models.items():
                if model:
                    try: row[f"hmm_loglik_class_{cls}"] = model.score(seq) / n_times_hmm
                    except: row[f"hmm_loglik_class_{cls}"] = -9999.0
                else: row[f"hmm_loglik_class_{cls}"] = -9999.0

            for c_idx, c_name in enumerate(target_ch_names):
                psd_ch = psds_all[i, c_idx, :]
                for band, (fmin, fmax) in BANDS.items():
                    f_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                    if np.any(f_idx):
                        val = psd_ch[f_idx].mean()
                        row[f"{c_name}_{band}_log_power"] = np.log10(val + 1e-15)
                    else:
                        row[f"{c_name}_{band}_log_power"] = -15.0

                erp_ch = erp_full[i, c_idx, :]
                row[f"{c_name}_mean_amp"] = erp_ch.mean()
                row[f"{c_name}_peak_amp"] = erp_ch.max()
                row[f"{c_name}_peak_lat"] = erp_times[erp_ch.argmax()]

            if i < split_idx:
                train_rows.append(row)
            else:
                test_rows.append(row)

    def save(rows, path):
        if not rows: return
        df = pd.DataFrame(rows)
        cols = ['subject_id', 'trial_id', 'label'] + sorted([c for c in df.columns if c not in ['subject_id', 'trial_id', 'label']])
        df[cols].to_csv(path, index=False)
        print(f"-> Finished: {path} | Shape: {df.shape}")
    save(train_rows, out_train)
    save(test_rows, out_test)

if __name__ == "__main__":
    extract_features_hmm('./data/0.5-30', 'train_features_hmm_2.csv', 'test_features_hmm_2.csv')