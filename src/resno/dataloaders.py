"""
This file contains the dataloaders for the explicit form RDEs and real-world data.
"""

from nvidia.dali.plugin.jax.iterator import DALIGenericIterator
from nvidia.dali import fn
from nvidia.dali.plugin.jax import data_iterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from data.driver_and_solution_info import RDE, rde_locations
from data.rdes import save_rde_and_driver_paths


def make_pipeline(
    batch_size: int,
    sample_name: str,
    label_name: str,
    file_root: str,
    file_filter_x: str,
    file_filter_y: str,
    shuffle_after_epoch: bool,
    seed: int,
) -> DALIGenericIterator:
    @data_iterator(
        output_map=[sample_name, label_name],
        size=100,
        auto_reset=False,
        last_batch_policy=LastBatchPolicy.FILL,
    )
    def pipeline():
        sample = fn.readers.numpy(device="cpu", file_root=file_root, file_filter=file_filter_x, name=sample_name, shuffle_after_epoch=shuffle_after_epoch, seed=seed)
        label = fn.readers.numpy(device="cpu", file_root=file_root, file_filter=file_filter_y, name=label_name, shuffle_after_epoch=shuffle_after_epoch, seed=seed)
        return sample, label

    return pipeline(batch_size=batch_size)


def get_eigenworms_pipelines(batch_size: int, seed: int) -> tuple[DALIGenericIterator, DALIGenericIterator, DALIGenericIterator]:
    train_pipeline = make_pipeline(batch_size, "worms", "labels", "data/eigenworms/train", "X_train_*.npy", "y_train_*.npy", True, seed)
    val_pipeline = make_pipeline(batch_size, "worms", "labels", "data/eigenworms/val", "X_val_*.npy", "y_val_*.npy", False, seed)
    test_pipeline = make_pipeline(batch_size, "worms", "labels", "data/eigenworms/test", "X_test_*.npy", "y_test_*.npy", False, seed)

    return train_pipeline, val_pipeline, test_pipeline


def get_rde_pipelines(batch_size: int, rough_paths: bool, rde: RDE, seed: int) -> tuple[DALIGenericIterator, DALIGenericIterator, DALIGenericIterator]:
    if rough_paths:
        train_pipeline = make_pipeline(
            batch_size,
            "solution_path",
            "driving_path",
            f"{rde_locations[rde]}/train/rough_paths",
            "y_rough_solution_*.npy",
            "X_rough_driver_*.npy",
            True,
            seed,
        )
        val_pipeline = make_pipeline(
            batch_size,
            "solution_path",
            "driving_path",
            f"{rde_locations[rde]}/val/rough_paths",
            "y_rough_solution_*.npy",
            "X_rough_driver_*.npy",
            False,
            seed,
        )
        test_pipeline = make_pipeline(
            batch_size,
            "solution_path",
            "driving_path",
            f"{rde_locations[rde]}/test/rough_paths",
            "y_rough_solution_*.npy",
            "X_rough_driver_*.npy",
            False,
            seed,
        )
    else:
        train_pipeline = make_pipeline(
            batch_size,
            "solution_path",
            "driving_path",
            f"{rde_locations[rde]}/train/paths",
            "y_solution_*.npy",
            "X_driver_*.npy",
            True,
            seed,
        )
        val_pipeline = make_pipeline(
            batch_size,
            "solution_path",
            "driving_path",
            f"{rde_locations[rde]}/val/paths",
            "y_solution_*.npy",
            "X_driver_*.npy",
            False,
            seed,
        )
        test_pipeline = make_pipeline(
            batch_size,
            "solution_path",
            "driving_path",
            f"{rde_locations[rde]}/test/paths",
            "y_solution_*.npy",
            "X_driver_*.npy",
            False,
            seed,
        )

    return train_pipeline, val_pipeline, test_pipeline


if __name__ == "__main__":
    batch_size = 1
    train_iterator, val_iterator, test_iterator = get_rde_pipelines(batch_size, rough_paths=False, rde=RDE.fOU, seed=42)

    for epoch in range(1):
        print(next(train_iterator))
        print("--------------------------------")
