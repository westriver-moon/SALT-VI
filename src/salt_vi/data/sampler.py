import numpy as np
from torch.utils.data.sampler import Sampler

def validate_identity_batch_config(batch_size, num_pos, number_of_identities):
    """Validate the PK-batch contract and return identities per batch."""
    batch_size = int(batch_size)
    num_pos = int(num_pos)
    number_of_identities = int(number_of_identities)
    if batch_size <= 0 or num_pos <= 0:
        raise ValueError(f"batch_size and num_pos must be positive; got {batch_size}, {num_pos}")
    if batch_size % num_pos != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by num_pos ({num_pos}) for PK sampling"
        )
    identities_per_batch = batch_size // num_pos
    if identities_per_batch > number_of_identities:
        raise ValueError(
            f"PK sampling requests {identities_per_batch} identities per batch, but only "
            f"{number_of_identities} are available"
        )
    return identities_per_batch


def GenIdx(train_color_label, train_thermal_label):
    """Return sample positions keyed by identity label for both modalities.

    Labels need not be dense or start at zero.  ReID datasets commonly remap
    labels, but making that an implementation precondition caused IndexError
    for otherwise valid identity annotations.
    """
    train_color_label = np.asarray(train_color_label)
    train_thermal_label = np.asarray(train_thermal_label)
    color_pos = {
        label: np.flatnonzero(train_color_label == label)
        for label in np.unique(train_color_label)
    }
    thermal_pos = {
        label: np.flatnonzero(train_thermal_label == label)
        for label in np.unique(train_thermal_label)
    }
    if set(color_pos) != set(thermal_pos):
        missing_color = sorted(set(thermal_pos) - set(color_pos))
        missing_thermal = sorted(set(color_pos) - set(thermal_pos))
        raise ValueError(
            "RGB and IR identity sets are inconsistent; "
            f"missing from RGB={missing_color}, missing from IR={missing_thermal}"
        )

    return color_pos, thermal_pos

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        validate_identity_batch_config(int(batchSize) * int(num_pos), num_pos, self.n_classes)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return len(self.index1)


class AutoReplaceIdentitySampler(Sampler):
    """Identity sampler that replaces only when an ID has too few samples."""

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        validate_identity_batch_config(int(batchSize) * int(num_pos), num_pos, self.n_classes)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        index1 = []
        index2 = []
        for _ in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for label in batch_idx:
                color_pool = color_pos[label]
                thermal_pool = thermal_pos[label]
                sample_color = np.random.choice(
                    color_pool, num_pos, replace=len(color_pool) < num_pos
                )
                sample_thermal = np.random.choice(
                    thermal_pool, num_pos, replace=len(thermal_pool) < num_pos
                )
                index1.extend(sample_color)
                index2.extend(sample_thermal)

        self.index1 = np.asarray(index1)
        self.index2 = np.asarray(index2)
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return len(self.index1)
