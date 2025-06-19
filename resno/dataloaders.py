from nvidia.dali import fn
from nvidia.dali.plugin.jax import data_iterator


batch_size = 32


@data_iterator(output_map=["worms", "labels"])
def eigenworms_train_pipeline():
    worms = fn.readers.numpy(device="cpu", file_root="data/eigenworms/train", file_filter="X_train_*.npy", name="worms_reader", shuffle_after_epoch=True, seed=42)
    labels = fn.readers.numpy(device="cpu", file_root="data/eigenworms/train", file_filter="y_train_*.npy", name="labels_reader", shuffle_after_epoch=True, seed=42)
    return worms, labels


@data_iterator(output_map=["worms", "labels"])
def eigenworms_val_pipeline():
    worms = fn.readers.numpy(device="cpu", file_root="data/eigenworms/val", file_filter="X_val_*.npy", name="worms_reader", shuffle_after_epoch=False, seed=42)
    labels = fn.readers.numpy(device="cpu", file_root="data/eigenworms/val", file_filter="y_val_*.npy", name="labels_reader", shuffle_after_epoch=False, seed=42)
    return worms, labels


# Not so shuffled due to order in the file names?
@data_iterator(output_map=["worms", "labels"])
def eigenworms_test_pipeline():
    worms = fn.readers.numpy(device="cpu", file_root="data/eigenworms/test", file_filter="X_test_*.npy", name="worms_reader", shuffle_after_epoch=False, seed=42)
    labels = fn.readers.numpy(device="cpu", file_root="data/eigenworms/test", file_filter="y_test_*.npy", name="labels_reader", shuffle_after_epoch=False, seed=42)
    return worms, labels


train_iterator = eigenworms_train_pipeline(batch_size=batch_size)
val_iterator = eigenworms_val_pipeline(batch_size=batch_size)
test_iterator = eigenworms_test_pipeline(batch_size=batch_size)

if __name__ == "__main__":
    for epoch in range(5):
        print(next(train_iterator)["labels"])
        print(next(val_iterator)["labels"])
        print(next(test_iterator)["labels"])
        print("--------------------------------")

# @data_iterator(output_map=["worms", "labels"])
# def balls_pipeline():
#     worms = fn.readers.numpy(device="cpu", files=["data/eigenworms/old/X_train.npy"], name="worms_reader", random_shuffle=False, seed=42)
#     labels = fn.readers.numpy(device="cpu", files=["data/eigenworms/old/y_train.npy"], name="labels_reader", random_shuffle=False, seed=42)
#     return worms, labels


# iterator2 = balls_pipeline(batch_size=1)
