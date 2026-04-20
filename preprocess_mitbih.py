# preprocess_mitbih.py
import os
import numpy as np
import wfdb
from wfdb import processing
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

#  Configuration
DATA_DIR = "data/raw/mitbih"
OUTPUT_DIR = "data/processed"
SPLIT_DIR = "data/splits"

FS = 360
WINDOW_SEC = 10
WINDOW_SIZE = FS * WINDOW_SEC  # 3600
STEP_SIZE = WINDOW_SIZE  # non-overlapping windows


# MIT-BIH record list
RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]

NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
IGNORE_SYMBOLS = {"|", "~", "!", "+", "[", "]", '"', "x"}

# Utility functions

# Set up directories for processed data and splits
def ensure_dir():   
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)

# signal processing functions
# Apply a bandpass Butterworth filter to the signal to remove noise and baseline wander
def bandpass_filter(sig, fs=360, low=0.5, high=40, order=3):
    nyq = 0.5 * fs
    low_cut = low / nyq
    high_cut = high / nyq
    b, a = butter(order, [low_cut, high_cut], btype="band")
    return filtfilt(b, a, sig)

def normalize_window(window):
    mean_val = np.mean(window)
    std_val = np.std(window)
    if std_val < 1e-8:
        return window - mean_val
    return (window - mean_val) / std_val

def choose_label(record):
    # Implementation for choosing label based on record
    if hasattr(record, "sig_name") and record.sig_name is not None:
        for i, name in enumerate(record.sig_name):
            if name.upper() == "MLII":
                return i
    return 0  # normal

# annotation and labeling functions
def get_window_annotation_symbols(ann, start, end):
    symbols = []
    for i in range(len(ann.sample)):
        sample_idx = ann.sample[i]
        if start <= sample_idx < end:
            symbols.append(ann.symbol[i])
    return symbols

def label_window(annotation_symbols):
    filtered_symbols = []
    for sym in annotation_symbols:
        if sym in IGNORE_SYMBOLS:
            continue
        filtered_symbols.append(sym)

    if len(filtered_symbols) == 0:
        return None
    
    for sym in filtered_symbols:
        if sym not in NORMAL_SYMBOLS:
            return 1  # anomalous
    return 0  # normal

# Normalize a window by subtracting the mean and dividing by the standard deviation
def detect_r_peaks(window, fs=360):
    try: 
        xqrs = processing.XQRS(sig=window, fs=fs)
        xqrs.detect()
        r_peaks = xqrs.qrs_inds
    except Exception:
        r_peaks = np.array([], dtype=int)
    return r_peaks

def compute_r_peak_features(r_peaks, fs=360):
    peak_count = len(r_peaks)
    if peak_count > 1:
        rr_intervals = np.diff(r_peaks) / fs
        mean_rr = float(np.mean(rr_intervals))
        std_rr = float(np.std(rr_intervals))
        min_rr = float(np.min(rr_intervals))
        max_rr = float(np.max(rr_intervals))
    else:
        mean_rr = 0.0
        std_rr = 0.0
        min_rr = 0.0
        max_rr = 0.0

    return np.array([peak_count, mean_rr, std_rr, min_rr, max_rr], dtype=np.float32)

# record processing function
def process_record(record_name, data_dir):
    print(f"Processing record {record_name}...")

    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    lead_idx = choose_label(record)
    signal = record.p_signal[:, lead_idx]   # selected channel
    signal = bandpass_filter(signal, fs=FS)

    X_record = []
    y_record = []
    peak_features_record = []
    metadata_record = []

    for start_idx in range(0, len(signal) - WINDOW_SIZE + 1, STEP_SIZE):
        end_idx = start_idx + WINDOW_SIZE
        window = signal[start_idx:end_idx]

        # R-peak detection
        r_peaks = detect_r_peaks(window, fs=FS)
        r_peak_features = compute_r_peak_features(r_peaks, fs=FS)

        annotation_symbols = get_window_annotation_symbols(ann, start_idx, end_idx)
        label = label_window(annotation_symbols)

        if label is None:
            continue  # skip windows with only ignored symbols

        window = normalize_window(window)

        X_record.append(window.astype(np.float32))
        y_record.append(label)
        peak_features_record.append(r_peak_features)
        metadata_record.append([
            record_name,
            start_idx,
            end_idx,
            len(r_peaks)
        ])
    X_record = np.array(X_record, dtype=np.float32)
    y_record = np.array(y_record, dtype=np.int32)
    peak_features_record = np.array(peak_features_record, dtype=np.float32)

    print(f" Kept windows: {len(X_record)}")
    if len(y_record) > 0:
        print(f"   Noramal: {np.sum(y_record == 0)} | Anomalous: {np.sum(y_record == 1)}")
    return X_record, y_record, peak_features_record, metadata_record

# helpers for splitting and saving
def save_record_split(record_names, filepath):
    with open(filepath, "w") as f:
        for record_name in record_names:
            f.write(record_name + "\n")

def load_and_process_split(record_names):
    X_all = []
    y_all = []
    peak_features_all = []
    metadata_all = []

    for record_name in record_names:
        X_record, y_record, peak_features_record, metadata_record = process_record(record_name, DATA_DIR)
        
        if len(X_record) > 0:
            X_all.append(X_record)
            y_all.append(y_record)
            peak_features_all.append(peak_features_record)
            metadata_all.extend(metadata_record)
    if len(X_all) == 0:
        return (
            np.array([], dtype = np.float32),
            np.array([], dtype = np.int32),
            np.array([], dtype = np.float32),
            []
        )

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    peak_features_all = np.concatenate(peak_features_all, axis=0)

    return X_all, y_all, peak_features_all, metadata_all

def save_metadata_csv(metadata, filepath):
    import csv
    with open(filepath, "w") as f:
        f.write("record_name,start_idx,end_idx,r_peak_count\n")
        for row in metadata:
            record_name, start_idx, end_idx, r_peak_count = row
            f.write(f"{record_name},{start_idx},{end_idx},{r_peak_count}\n")

def main():
    ensure_dir()

    train_records, temp_records = train_test_split(RECORDS, test_size=0.3, random_state=42)
    val_records, test_records = train_test_split(temp_records, test_size=0.5, random_state=42)

    save_record_split(train_records, os.path.join(SPLIT_DIR, "train_records.txt"))
    save_record_split(val_records, os.path.join(SPLIT_DIR, "val_records.txt"))
    save_record_split(test_records, os.path.join(SPLIT_DIR, "test_records.txt"))

    print("Train records:", train_records)
    print("Validation records:", val_records)
    print("Test records:", test_records)

    X_train, y_train, peak_features_train, metadata_train = load_and_process_split(train_records)
    X_val, y_val, peak_features_val, metadata_val = load_and_process_split(val_records)
    X_test, y_test, peak_features_test, metadata_test = load_and_process_split(test_records)

    print(f"\nFinal dataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, peak_features_train: {peak_features_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}, peak_features_val: {peak_features_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}, peak_features_test: {peak_features_test.shape}")

    if len(y_train) > 0:
        print(f"\nTrain class distribution: Normal (0): {np.sum(y_train == 0)}, Anomalous (1): {np.sum(y_train == 1)}")
    if len(y_val) > 0:
        print(f"Validation class distribution: Normal (0): {np.sum(y_val == 0)}, Anomalous (1): {np.sum(y_val == 1)}")
    if len(y_test) > 0:
        print(f"Test class distribution: Normal (0): {np.sum(y_test == 0)}, Anomalous (1): {np.sum(y_test == 1)}")

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)

    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)


    np.save(os.path.join(OUTPUT_DIR, "peak_features_train.npy"), peak_features_train)
    np.save(os.path.join(OUTPUT_DIR, "peak_features_val.npy"), peak_features_val)
    np.save(os.path.join(OUTPUT_DIR, "peak_features_test.npy"), peak_features_test)

    save_metadata_csv(metadata_train, os.path.join(OUTPUT_DIR, "metadata_train.csv"))
    save_metadata_csv(metadata_val, os.path.join(OUTPUT_DIR, "metadata_val.csv"))
    save_metadata_csv(metadata_test, os.path.join(OUTPUT_DIR, "metadata_test.csv"))

    print("\nSaved processed ECG windows, labels, R-peak features, and metadata.")

if __name__ == "__main__":
    main()