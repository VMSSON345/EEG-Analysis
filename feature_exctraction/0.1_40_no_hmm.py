import mne
import pandas as pd
import numpy as np
import glob
import os
import re
from mne.time_frequency import psd_array_welch

def extract_features_no_hmm(base_data_path: str, output_csv: str):
    EXCLUDE_SUBJECTS = ['sub-25', 'sub-16', 'sub-23', 'sub-35']

    BANDS = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 40.0)
    }
    ERP_WINDOW = (0.25, 0.6)

    search_path = os.path.join(base_data_path, '*_cleaned.fif')
    epoch_files = glob.glob(search_path)

    if not epoch_files:
        print("Do not find any .fif files.")
        return
    print(f"Found {len(epoch_files)} files.")

    all_rows = []

    for file_path in epoch_files:
        filename = os.path.basename(file_path)
        match = re.search(r'(sub[_-]?\d+|subject[_-]?\d+)', filename.lower())
        subject_id = match.group(0) if match else filename.split('_')[0]

        # Lọc Subject
        norm_id = subject_id.replace('_', '-').lower()
        if any(bad in norm_id for bad in EXCLUDE_SUBJECTS):
            print(f"[SKIP]: {subject_id}")
            continue

        # Đọc dữ liệu
        epochs = mne.read_epochs(file_path, preload=True, verbose='ERROR')
        print(f"Proccessing: {subject_id}")

        try:
            epochs_subset = epochs['Standard', 'Target']
        except KeyError: continue

        target_ch_names = epochs_subset.ch_names
        sfreq = epochs.info['sfreq']
        n_trials = len(epochs_subset)

        # Nhãn: 0=Standard, 1=Target
        event_id_map = epochs.event_id
        def get_label(code):
            if code == event_id_map.get('Standard'): return 0
            if code == event_id_map.get('Target'): return 1
            return -1
        labels = np.array([get_label(c) for c in epochs_subset.events[:, 2]])

        full_data = epochs_subset.get_data() # (n_trials, n_ch, n_times)

        t_idx_start, t_idx_end = epochs_subset.time_as_index(ERP_WINDOW)
        erp_full = full_data[:, :, t_idx_start:t_idx_end]
        erp_times = epochs_subset.times[t_idx_start:t_idx_end]

        psds_all, freqs = psd_array_welch(
            full_data, sfreq=sfreq, fmin=0.1, fmax=40.0, n_fft=256, verbose='ERROR'
        )

        for i in range(n_trials):
            feature_row = {
                'subject_id': subject_id,
                'trial_id': i,
                'label': labels[i]
            }

            for c_idx, c_name in enumerate(target_ch_names):
                psd_ch = psds_all[i, c_idx, :]
                for band, (fmin, fmax) in BANDS.items():
                    freq_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                    if np.any(freq_idx):
                        val = psd_ch[freq_idx].mean()
                        feature_row[f"{c_name}_{band}_log_power"] = np.log10(val + 1e-15)
                    else:
                        feature_row[f"{c_name}_{band}_log_power"] = -15.0

                erp_ch = erp_full[i, c_idx, :]
                feature_row[f"{c_name}_mean_amp"] = erp_ch.mean()
                feature_row[f"{c_name}_peak_amp"] = erp_ch.max()
                feature_row[f"{c_name}_peak_lat"] = erp_times[erp_ch.argmax()]

            all_rows.append(feature_row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        cols = ['subject_id', 'trial_id', 'label'] + sorted([c for c in df.columns if c not in ['subject_id', 'trial_id', 'label']])
        df = df[cols]
        df.to_csv(output_csv, index=False)
        print(f"Finish: {output_csv} | Shape: {df.shape}")

if __name__ == "__main__":
    extract_features_no_hmm('./data/0.1-40', 'features.csv')   