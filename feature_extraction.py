import mne
import pandas as pd
import numpy as np
import glob
import os
import re

def extract_features(base_data_path: str, output_csv: str):
    """
    Tìm tất cả các tệp epoch đã làm sạch, trích xuất đặc trưng PSD 
    (Delta, Theta, Alpha, Beta) và đặc trưng miền thời gian (ERP)
    và lưu vào một tệp CSV duy nhất.
    """
    
    # Các dải tần số
    BANDS = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0)
    }
    
    # đặc trưng miền thời gian
    ERP_WINDOW = (0.25, 0.6) # 250ms đến 600ms

    search_path = os.path.join(base_data_path, '*_cleaned.fif')
    epoch_files = glob.glob(search_path)
    
    if not epoch_files:
        print(f"Không tìm thấy tệp .fif nào tại: {search_path}")
        return

    print(f"Tìm thấy {len(epoch_files)} tệp. Bắt đầu xử lý...")

    all_features_list = []

    for file_path in epoch_files:
        epochs = mne.read_epochs(file_path, preload=True, verbose='ERROR')
        
        filename = os.path.basename(file_path)
        match = re.search(r'(sub[_-]?\d+|subject[_-]?\d+)', filename.lower())
        subject_id = match.group(0) if match else filename.split('_')[0]
            
        print(f"  Đang xử lý: {subject_id}")

        epochs_subset = epochs['Standard', 'Target', 'Distractor']

        event_id_map = epochs.event_id

        def get_label(event_code):
            if event_code == event_id_map['Standard']:
                return 0
            elif event_code == event_id_map['Target']:
                return 1
            elif event_code == event_id_map['Distractor']:
                return 2
            return -1
            
        labels = [get_label(event_code) for event_code in epochs_subset.events[:, 2]]
        
        ch_names = epochs_subset.ch_names
        
        # Trích xuất đặc trưng miền tần số PSD
        
        spectrum = epochs_subset.compute_psd(
            method='welch', 
            fmin=BANDS["delta"][0], 
            fmax=BANDS["beta"][1], 
            n_fft=256, 
            average='mean', 
            verbose='WARNING'
        )
        psds = spectrum.get_data()
        freqs = spectrum.freqs
        
        band_powers = {}
        for band, (fmin, fmax) in BANDS.items():
            freq_indices = np.logical_and(freqs >= fmin, freqs <= fmax)
            raw_power = psds[:, :, freq_indices].mean(axis=2)
            band_powers[band] = np.log10(raw_power)
            
        
        # Trích xuất đặc trưng miền thời gian ERP
        
        t_idx_start, t_idx_end = epochs_subset.time_as_index(ERP_WINDOW)
        
        # Lấy dữ liệu thô (đã baseline) trong cửa sổ này
        # Shape: (n_epochs, n_channels, n_times_in_window)
        erp_data_window = epochs_subset.get_data(picks='eeg')[:, :, t_idx_start:t_idx_end].copy()
        
        # Lấy mốc thời gian trong cửa sổ (để tính latency)
        window_times = epochs_subset.times[t_idx_start:t_idx_end]
        
        # Tính toán đặc trưng
        mean_amps = erp_data_window.mean(axis=2)
        peak_amps = erp_data_window.max(axis=2)
        
        argmax_indices = erp_data_window.argmax(axis=2)
        peak_latencies = window_times[argmax_indices]

        
        for i in range(len(epochs_subset)): # Lặp qua từng trial
            feature_row = {
                'subject_id': subject_id,
                'trial_id': i,
                'label': labels[i]
            }
            
            # PSD
            for band, powers in band_powers.items():
                for ch_idx, ch_name in enumerate(ch_names):
                    feature_name = f"{ch_name}_{band}_log_power"
                    feature_row[feature_name] = powers[i, ch_idx]
            
            # ERP
            for ch_idx, ch_name in enumerate(ch_names):
                feature_row[f"{ch_name}_mean_amp"] = mean_amps[i, ch_idx]
                feature_row[f"{ch_name}_peak_amp"] = peak_amps[i, ch_idx]
                feature_row[f"{ch_name}_peak_lat"] = peak_latencies[i, ch_idx]
            
            all_features_list.append(feature_row)

    if not all_features_list:
        print("Không có dữ liệu nào được trích xuất.")
        return
        
    final_df = pd.DataFrame(all_features_list)
    
    cols = ['subject_id', 'trial_id', 'label'] + \
           sorted([col for col in final_df.columns if col not in ['subject_id', 'trial_id', 'label']])
    final_df = final_df[cols]

    final_df.to_csv(output_csv, index=False)
    print(f"\nHoàn tất! Đã lưu đặc trưng vào: {output_csv}")
    print(f"Tổng số đặc trưng: {len(final_df.columns) - 3}")
    print(f"Tổng số hàng (trials): {len(final_df)}")

# --- Bắt đầu chạy script ---
if __name__ == "__main__":
    PATH_TO_DATA = './data' 
    OUTPUT_CSV_FILE = './data/features.csv'
    
    extract_features(PATH_TO_DATA, OUTPUT_CSV_FILE)