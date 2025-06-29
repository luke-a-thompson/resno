from scipy.io import arff
import pandas as pd
import numpy as np
import numpy.typing as npt
from pathlib import Path
import urllib.request
import urllib.error
import zipfile
import os
from tqdm import tqdm
from typing import Literal, Tuple


def parse_row(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    - If s is a string: strip newlines, literal_eval, then convert to (6xT) float array.
    - If s is a NumPy array with a structured dtype: unpack its field-values into a 1D array of floats.
    - Otherwise (e.g. Python list/tuple or plain np.ndarray): just cast to float.
    """
    # 2a) Structured dtype (i.e. s.dtype.names is a non‐None tuple of field names)
    if s.dtype.names is not None and any(name.startswith("att") for name in s.dtype.names):
        # s is something like: array((1.2, 3.4, 5.6, …), dtype=[('att17960','<f8'), …, ('att17984','<f8')])
        # We can call s.tolist() to get a tuple of all field‐values,
        # then turn that into a float array:
        return np.array(s.tolist(), dtype=float)

    # 2b) Plain numeric array (no named fields). Just cast to float:
    return s.astype(float)


def save_data_by_worm(X: npt.NDArray[np.float64], y: npt.NDArray[np.int_], output_dir: Path, split_name: str) -> None:
    """
    Save each worm's data and label as separate .npy files.

    Args:
        X: Data array of shape (n_worms, n_dims, n_timepoints)
        y: Labels array of shape (n_worms,)
        output_dir: Directory to save the files
        split_name: Name of the split (e.g., "train", "val", "test")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(X.shape[0]):
        worm_id = str(i).zfill(3)
        X_path = output_dir / f"X_{split_name}_{worm_id}.npy"
        y_path = output_dir / f"y_{split_name}_{worm_id}.npy"
        np.save(X_path, X[i])
        np.save(y_path, y[i])
    print(f"Saved {X.shape[0]} worms to {output_dir} for split '{split_name}'")


def eigenworms_to_numpy(val_ratio: float, seed: int) -> None:
    """
    Convert the EigenWorms dataset to individual .npy files for each worm's features (X)
    and labels (y), and create a validation split.

    Args:
        val_ratio: Proportion of training data to use for validation (0.0 to 1.0)
        seed: Random seed for reproducible validation splits
    """
    train_dl_path = Path("data_raw") / "eigenworms" / "EigenWorms_TRAIN.arff"
    test_dl_path = Path("data_raw") / "eigenworms" / "EigenWorms_TEST.arff"
    output_dir = Path("data/eigenworms")

    # Clear existing data
    if output_dir.exists():
        for split in ["train", "val", "test"]:
            split_dir = output_dir / split
            if split_dir.exists():
                for f in split_dir.glob("*.npy"):
                    f.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process training data and create validation split
    data, _ = arff.loadarff(train_dl_path)
    df = pd.DataFrame(data)
    # Extract features (X)
    arrays_2d = df["eigenWormMultivariate_attribute"].apply(parse_row)
    X_train_full: npt.NDArray[np.float64] = np.stack(arrays_2d.to_list(), axis=0)
    # Extract labels (y)
    y_train_full = df["target"].str.decode("utf-8").astype(int).to_numpy()

    # Create validation split
    n_worms = X_train_full.shape[0]
    n_val = int(n_worms * val_ratio)
    np.random.seed(seed)
    val_indices = np.random.choice(n_worms, size=n_val, replace=False)
    train_indices = np.setdiff1d(np.arange(n_worms), val_indices)

    X_train, y_train = X_train_full[train_indices], y_train_full[train_indices]
    X_val, y_val = X_train_full[val_indices], y_train_full[val_indices]

    # Save training and validation data
    save_data_by_worm(X_train, y_train, output_dir / "train", "train")
    save_data_by_worm(X_val, y_val, output_dir / "val", "val")

    # Process test data
    data, _ = arff.loadarff(test_dl_path)
    df = pd.DataFrame(data)
    # Extract features (X)
    arrays_2d = df["eigenWormMultivariate_attribute"].apply(parse_row)
    X_test: npt.NDArray[np.float64] = np.stack(arrays_2d.to_list(), axis=0)
    # Extract labels (y)
    y_test = df["target"].str.decode("utf-8").astype(int).to_numpy()

    # Save test data
    save_data_by_worm(X_test, y_test, output_dir / "test", "test")


def download_eigenworms_dataset(url: str = "https://www.timeseriesclassification.com/aeon-toolkit/EigenWorms.zip") -> None:
    """
    Download EigenWorms dataset zip file and extract _TEST and _TRAIN arff files to data_raw/eigenworms.
    """
    # Create target directory
    target_dir = Path("data_raw/eigenworms")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download the zip file with progress bar
    print(f"Downloading {url}...")
    zip_path = target_dir / "temp_eigenworms.zip"
    try:
        with urllib.request.urlopen(url) as response:
            # Get file size for progress bar
            file_size = int(response.headers.get("Content-Length", 0))

            # Save zip file temporarily with progress bar
            with open(zip_path, "wb") as f, tqdm(total=file_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
                while True:
                    chunk = response.read(8192)  # Read in 8KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    except urllib.error.URLError as e:
        print(f"Error downloading file: {e}")
        return
    finally:
        if zip_path.exists():
            zip_path.unlink()
            print("Cleaned up temporary zip file")

    # Extract only the _TEST and _TRAIN arff files
    print("Extracting arff files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            if base_name in ["EigenWorms_TEST.arff", "EigenWorms_TRAIN.arff"]:
                extracted_path = target_dir / base_name
                with zip_ref.open(file_name) as source, open(extracted_path, "wb") as target:
                    target.write(source.read())
                print(f"Extracted: {extracted_path}")

    print(f"Dataset extracted to {target_dir}")


if __name__ == "__main__":
    if not Path("data_raw/eigenworms/EigenWorms_TRAIN.arff").exists() or not Path("data_raw/eigenworms/EigenWorms_TEST.arff").exists():
        download_eigenworms_dataset()

    eigenworms_to_numpy(val_ratio=0.2, seed=42)
