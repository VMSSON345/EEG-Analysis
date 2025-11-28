# preprocess_all.py
import os
import sys
import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Tắt GUI để chạy nền
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne_icalabel import label_components
import time

# Đường dẫn log
LOG_PATH = "log.txt"

def log_message(msg):
    """Ghi log vào file và in ra màn hình"""
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def preprocess_subject(subject_id, input_dir, output_dir):
    sub_str = f"sub-{subject_id:02d}"
    start_time = time.time()
    log_message(f"\n{'='*60}")
    log_message(f"[START] Xử lý {sub_str} lúc {time.ctime()}")
    log_message(f"{'='*60}")

    try:
        # --- Cell 0: Nhập thư viện & nạp dữ liệu ---
        vhdr_path = os.path.join(input_dir, sub_str, f"{sub_str}_task-oddball_eeg.vhdr")
        if not os.path.exists(vhdr_path):
            log_message(f"⚠️  {sub_str}: Không tìm thấy file {vhdr_path}")
            return False

        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
        log_message(f"[Cell 0] Đã nạp dữ liệu thô. Tổng số kênh: {len(raw.ch_names)}")

        # --- Cell 1: Lựa chọn & chuẩn bị kênh ---
        ref_channels = ['TP9 LEFT EAR', 'TP10 RIGHT EAR']
        edge_channels = [
            'FT9', 'FT10', 'F9', 'F10', 'FFT9h', 'FFT10h', 
            'FTT9h', 'FTT10h', 'TTP7h', 'TTP8h', 'TPP9h', 'TPP10h',
            'P9', 'P10', 'PO9', 'PO10', 'O9', 'O10', 'POO9h', 'POO10h'
        ]
        eog_channels = ['AFp1', 'AFp2']

        raw_clean = raw.copy()
        channels_to_drop = [ch for ch in ref_channels + edge_channels if ch in raw_clean.ch_names]
        raw_clean.drop_channels(channels_to_drop)
        log_message(f"[Cell 1] Đã loại {len(channels_to_drop)} kênh. Còn lại: {len(raw_clean.ch_names)} kênh.")

        existing_eog = [ch for ch in eog_channels if ch in raw_clean.ch_names]
        if existing_eog:
            raw_clean.set_channel_types({ch: 'eog' for ch in existing_eog})
            log_message(f"Đã đặt {existing_eog} làm kênh EOG.")
        else:
            log_message("⚠️ Không tìm thấy kênh EOG!")

        # --- Cell 2: CAR + Montage ---
        log_message("[Cell 2] Đang áp dụng Common Average Reference (CAR)...")
        raw_clean.set_eeg_reference('average', projection=False)
        log_message("Đang gán montage 'standard_1005'...")
        montage = mne.channels.make_standard_montage('standard_1005')
        raw_clean.set_montage(montage, on_missing='warn')
        log_message("✅ Hoàn thành chuẩn hóa không gian kênh.")

        # --- Cell 3: Lọc tần số chính (0.1–40 Hz) ---
        log_message("[Cell 3] Đang lọc...")
        raw_clean.notch_filter(freqs=50, fir_design='firwin')
        raw_clean.filter(l_freq=0.1, h_freq=40.0, fir_design='firwin')
        log_message("✅ Lọc notch 50 Hz + band-pass 0.1–40 Hz hoàn tất.")

        # --- Cell 4: ICA (dùng extended infomax + dải 1–100 Hz cho ICLabel) ---
        raw_for_ica = raw_clean.copy().filter(l_freq=1.0, h_freq=100.0)
        ica = ICA(
            n_components=0.95,
            method='infomax',
            fit_params=dict(extended=True),
            random_state=97,
            max_iter=800
        )
        log_message("[Cell 4] Đang huấn luyện ICA với extended infomax...")
        ica.fit(raw_for_ica)
        log_message(f"✅ ICA hoàn tất. Số thành phần: {ica.n_components_}")

        # --- Cell 5: ICLabel + Loại nhiễu ---
        ica.labels_ = {}
        log_message("[Cell 5] Đang chạy ICLabel để phân loại thành phần...")
        ic_labels_dict = label_components(raw_for_ica, ica, method='iclabel')
        log_message(f"Kết quả phân loại (list): {ic_labels_dict['labels']}")

        exclude_labels = ['eog', 'muscle', 'ecg', 'line_noise', 'ch_noise', 'other']
        bad_indices = []
        for label in exclude_labels:
            if label in ica.labels_:
                bad_indices.extend(ica.labels_[label])
        ica.exclude = sorted(set(bad_indices))
        brain_indices = ica.labels_.get('brain', [])
        log_message(f"\n✅ Thành phần 'brain' giữ lại: {brain_indices}")
        log_message(f"❌ Thành phần bị loại: {ica.exclude}")

        # --- Cell 6: Áp dụng ICA & dọn dẹp ---
        log_message("[Cell 6] Đang áp dụng ICA để khử nhiễu...")
        ica.apply(raw_clean)
        eog_to_drop = [ch for ch in eog_channels if ch in raw_clean.ch_names]
        if eog_to_drop:
            raw_clean.drop_channels(eog_to_drop)
            log_message(f"Đã loại kênh EOG: {eog_to_drop}")
        log_message(f"✅ Số kênh EEG sạch cuối cùng: {len(raw_clean.ch_names)}")

        # --- Cell 7: Tạo epochs cho bài toán 2 lớp (Target vs Standard) ---
        event_id_map = {
            'Stimulus/S  5': 5,   # Standard
            'Stimulus/S  6': 6    # Target
        }
        log_message(f"Sẽ trích xuất các sự kiện: {event_id_map}")

        try:
            events, _ = mne.events_from_annotations(raw_clean, event_id=event_id_map)
        except ValueError as e:
            log_message("\n--- LỖI ---")
            log_message("Không tìm thấy sự kiện phù hợp. Kiểm tra tên marker trong file .vmrk!")
            raise e

        log_message(f"\nTìm thấy {len(events)} sự kiện. 5 sự kiện đầu:")
        log_message(str(events[:5]))

        epoch_event_id = {'Standard': 5, 'Target': 6}
        log_message(f"\nSử dụng Epoch IDs: {epoch_event_id}")

        tmin, tmax = -0.2, 0.8
        log_message("Đang tạo epochs...")
        epochs = mne.Epochs(
            raw_clean,
            events,
            epoch_event_id,
            tmin, tmax,
            baseline=(-0.2, 0),
            reject=None,
            preload=True
        )
        log_message(f"Đã tạo {len(epochs)} epochs ban đầu.")

        threshold_uv = 100.0
        epochs_clean = epochs.copy().drop_bad(reject={'eeg': threshold_uv * 1e-6})

        log_message(f"\nSố epochs ban đầu: {len(epochs)}")
        log_message(f"Số epochs sạch: {len(epochs_clean)}")
        log_message(f"Số epochs bị loại: {len(epochs) - len(epochs_clean)}")

        log_message("\nChi tiết theo điều kiện:")
        for condition in epoch_event_id.keys():
            try:
                n_start = len(epochs[condition])
                n_clean = len(epochs_clean[condition])
                log_message(f"  - {condition}: {n_start} → {n_clean} (loại {n_start - n_clean})")
            except KeyError:
                log_message(f"  - {condition}: Không có epochs nào!")

        if len(epochs_clean) == 0:
            log_message("❌ Không có epochs sạch để lưu!")
            return False

        # --- Cell 8: Trực quan hóa ERP cho 2 lớp (Target vs Standard) ---
        # Bỏ qua plot khi chạy hàng loạt (đã tắt GUI), nhưng bạn có thể lưu ảnh nếu cần

        # --- Cell 9: Lưu dữ liệu ---
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{sub_str}_cleaned-epo.fif")
        epochs_clean.save(save_path, overwrite=True)
        log_message(f"[Cell 9] ✅ Đã lưu epochs sạch vào: {save_path}")

        elapsed = time.time() - start_time
        log_message(f"[SUCCESS] {sub_str} hoàn thành trong {elapsed:.1f} giây.")
        return True

    except Exception as e:
        log_message(f"[ERROR] {sub_str} gặp lỗi: {str(e)}")
        return False

# --- Chạy cho 42 subject ---
if __name__ == "__main__":
    # === CẤU HÌNH ĐƯỜNG DẪN ===
                  
    INPUT_DIR = r"E:\UNIVERSITY\neurouScience\btl-EEG\preprocess\data\original"
    OUTPUT_DIR = r"E:\UNIVERSITY\neurouScience\btl-EEG\preprocess\zauto"
    
    # Xóa log cũ, tạo mới
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    log_message("🚀 BẮT ĐẦU TIỀN XỬ LÝ TOÀN BỘ 42 SUBJECT")
    log_message(f"Thư mục input: {INPUT_DIR}")
    log_message(f"Thư mục output: {OUTPUT_DIR}\n")

    success_count = 0
    for sid in range(1, 43):
        if preprocess_subject(sid, INPUT_DIR, OUTPUT_DIR):
            success_count += 1

    log_message(f"\n{'='*60}")
    log_message(f"🏁 HOÀN TẤT: {success_count}/42 subject thành công")
    log_message(f"📄 Xem chi tiết tại: {os.path.abspath(LOG_PATH)}")