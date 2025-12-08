import mne
import pandas as pd
import numpy as np
import glob
import os
import re
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from mne.time_frequency import psd_array_welch

def extract_features_hmm(base_data_path, output_dir):
    FIXED_TEST_SUBS = ['sub-03', 'sub-05', 'sub-06', 'sub-12', 'sub-24', 'sub-31', 'sub-39', 'sub-42']
    EXCLUDE_SUBS = ['sub-25', 'sub-16', 'sub-23', 'sub-35']
    HMM_CHANNELS = ['Pz', 'Cz', 'Fz', 'P3', 'P4']
    N_STATES = 5

    BANDS = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 40.0)
    }
    ERP_WINDOW = (0.25, 0.6)

    search_path = os.path.join(base_data_path, '*_cleaned.fif')
    all_files = glob.glob(search_path)

    train_files = []
    test_files = []
    sub_file_map = {}
    for f in all_files:
        fname = os.path.basename(f)
        match = re.search(r'(sub[_-]?\d+|subject[_-]?\d+)', fname.lower())
        raw_id = match.group(0) if match else fname.split('_')[0]

        num_part = re.findall(r'\d+', raw_id)[0]
        norm_id = f"sub-{int(num_part):02d}" # sub-3 -> sub-03

        if norm_id in EXCLUDE_SUBS:
            print(f"  [SKIP]: {norm_id} ({fname})")
            continue

        sub_file_map[norm_id] = f
        if norm_id in FIXED_TEST_SUBS:
            test_files.append(norm_id)
        else:
            train_files.append(norm_id)

    print(f"Train Subjects: {len(train_files)}")
    print(f"Test Subjects: {len(test_files)}")

    print(f"Training HMM Models...")

    scaler = StandardScaler()
    X_concat_0 = [] # Standard
    X_concat_1 = [] # Target
    lens_0 = []
    lens_1 = []
    raw_data_for_scaler = []

    for sub_id in train_files:
        fpath = sub_file_map[sub_id]
        epochs = mne.read_epochs(fpath, preload=True, verbose='ERROR')
        try: epochs_sub = epochs['Standard', 'Target']
        except: continue

        e_hmm = epochs_sub.copy().pick(HMM_CHANNELS)
        e_hmm.resample(100, verbose='ERROR')
        data = e_hmm.get_data().transpose(0, 2, 1) # (n_trials, n_times, n_ch)
        n_tr, n_ti, n_ch = data.shape
        eid = epochs.event_id
        lbls = [0 if epochs_sub.events[i,2] == eid['Standard'] else 1 for i in range(len(epochs_sub))]
        reshaped_data = data.reshape(-1, n_ch)
        raw_data_for_scaler.append(reshaped_data)
    scaler.fit(np.vstack(raw_data_for_scaler))
    del raw_data_for_scaler
    for sub_id in train_files:
        fpath = sub_file_map[sub_id]
        epochs = mne.read_epochs(fpath, preload=True, verbose='ERROR')
        try: epochs_sub = epochs['Standard', 'Target']
        except: continue
        e_hmm = epochs_sub.copy().pick(HMM_CHANNELS)
        e_hmm.resample(100, verbose='ERROR')
        data = e_hmm.get_data().transpose(0, 2, 1)

        eid = epochs.event_id
        lbls = [0 if epochs_sub.events[i,2] == eid['Standard'] else 1 for i in range(len(epochs_sub))]

        for i in range(len(data)):
            trial_scaled = scaler.transform(data[i])
            if lbls[i] == 0:
                X_concat_0.append(trial_scaled)
                lens_0.append(len(trial_scaled))
            else:
                X_concat_1.append(trial_scaled)
                lens_1.append(len(trial_scaled))

    print(f"-> Training HMMs (Standard: {len(lens_0)} trials, Target: {len(lens_1)} trials)...")

    hmm_std = hmm.GaussianHMM(n_components=N_STATES, covariance_type="diag", n_iter=10, random_state=42)
    hmm_std.fit(np.vstack(X_concat_0), lens_0)

    hmm_tar = hmm.GaussianHMM(n_components=N_STATES, covariance_type="diag", n_iter=10, random_state=42)
    hmm_tar.fit(np.vstack(X_concat_1), lens_1)

    hmm_models = {0: hmm_std, 1: hmm_tar}
    print("HMM Training Complete.")

    print("Extracting Features with HMM Scores...")

    train_rows = []
    test_rows = []
    all_targets = train_files + test_files

    for sub_id in all_targets:
        is_test = sub_id in test_files
        grp = "TEST" if is_test else "TRAIN"
        print(f"Processing: {sub_id} -> {grp}")

        fpath = sub_file_map[sub_id]
        epochs = mne.read_epochs(fpath, preload=True, verbose='ERROR')
        try: epochs_sub = epochs['Standard', 'Target']
        except: continue
        e_hmm = epochs_sub.copy().pick(HMM_CHANNELS)
        e_hmm.resample(100, verbose='ERROR')
        data_hmm = e_hmm.get_data().transpose(0, 2, 1)
        n_tr_hmm, n_ti_hmm, _ = data_hmm.shape

        hmm_scores = []
        for i in range(n_tr_hmm):
            trial_scaled = scaler.transform(data_hmm[i])
            scores = {}
            for cls, model in hmm_models.items():
                try: scores[cls] = model.score(trial_scaled) / n_ti_hmm
                except: scores[cls] = -9999.0
            hmm_scores.append(scores)

        full_data = epochs_sub.get_data()
        t_s, t_e = epochs_sub.time_as_index(ERP_WINDOW)
        erp_full = full_data[:, :, t_s:t_e]
        erp_times = epochs_sub.times[t_s:t_e]
        target_ch_names = epochs_sub.ch_names
        psds, freqs = psd_array_welch(full_data, sfreq=epochs.info['sfreq'],
                                      fmin=0.1, fmax=40.0, n_fft=256, verbose='ERROR')

        eid = epochs.event_id
        lbls = [0 if epochs_sub.events[i,2] == eid['Standard'] else 1 for i in range(len(epochs_sub))]
        for i in range(len(epochs_sub)):
            row = {'subject_id': sub_id, 'trial_id': i, 'label': lbls[i]}
            row['hmm_loglik_class_0'] = hmm_scores[i][0]
            row['hmm_loglik_class_1'] = hmm_scores[i][1]
            for c_idx, c_name in enumerate(target_ch_names):
                p_ch = psds[i, c_idx, :]
                for band, (bmi, bma) in BANDS.items():
                    f_idx = np.logical_and(freqs >= bmi, freqs <= bma)
                    val = p_ch[f_idx].mean() if np.any(f_idx) else 0
                    row[f"{c_name}_{band}_log_power"] = np.log10(val + 1e-15)
                # ERP
                e_ch = erp_full[i, c_idx, :]
                row[f"{c_name}_mean_amp"] = e_ch.mean()
                row[f"{c_name}_peak_amp"] = e_ch.max()
                row[f"{c_name}_peak_lat"] = erp_times[e_ch.argmax()]

            if is_test:
                test_rows.append(row)
            else:
                train_rows.append(row)

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    def save(rows, name):
        if not rows: return
        df = pd.DataFrame(rows)
        meta = ['subject_id', 'trial_id', 'label']
        feats = sorted([c for c in df.columns if c not in meta])
        df[meta + feats].to_csv(os.path.join(output_dir, name), index=False)
        print(f"Finished: {name} | Shape: {df.shape}")
    save(train_rows, 'sub_train_hmm.csv')
    save(test_rows, 'sub_test_hmm.csv')

if __name__ == "__main__":
    DATA_DIR = './data/0.1-40'
    OUTPUT_DIR = './features/0.1-40'

    extract_features_hmm(DATA_DIR, OUTPUT_DIR)