import math
import os
import sys
import shutil
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import nfp
from nfp.layers import RBFExpansion
from nfp.preprocessing.crystal_preprocessor import PymatgenPreprocessor
print(f"{pd.__version__ = }")
print(f"{np.__version__ = }")
print(f"{tf.__version__ = }")
print(f"{nfp.__version__ = }")

tqdm.pandas()


class AtomicNumberPreprocessor(PymatgenPreprocessor):
    def __init__(self, max_atomic_num=83, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.site_tokenizer = lambda x: Element(x).Z
        self._max_atomic_num = max_atomic_num

    @property
    def site_classes(self):
        return self._max_atomic_num


preprocessor = AtomicNumberPreprocessor(max_atomic_num=84)

if len(sys.argv) < 3:
    print(f"python {sys.argv[0]} <runid> <train_size_idx>")
    sys.exit()

# repeat the learning curve multiple times to get error bars on the test performance
runid = int(sys.argv[1])
# index of number of training examples to use from 100 to len(train) split by 10 in log space 
train_size_idx = int(sys.argv[2])
out_dir = f"run_{runid}/n_{train_size_idx}"
print(f"{out_dir = }")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Make a backup of the job submission script
shutil.copy(__file__, out_dir)

# Read the inputs
data_file = Path("inputs/preprocessed/scaled_inputs.p")
data = pd.read_pickle(data_file)
print(f"read {len(data)} structures from {data_file}")
print(data.head(2))
max_atomic_num = 84

composition_set = data.composition.isin(
    pd.Series(data.composition.unique()).sample(100, random_state=runid)
)
test_composition = data[composition_set]
train_composition = data[~composition_set]

train, test = train_test_split(
    train_composition,
    test_size=1500,
    random_state=runid,
    stratify=train_composition["type"],
)

# now subset to the specified number of training examples
orig_num_valid = len(train) * .05
orig_num_train = len(train) - orig_num_valid
train_size = np.log10(orig_num_train)
num_train = np.logspace(2, train_size, num=10, dtype=int)[train_size_idx]

# rather than sample at random, use a stratified split
# so each dataset is split evenly
#train = train.sample(num_train)
train, _ = train_test_split(
    train,
    train_size=num_train,
    stratify=train["type"],
)

# leave out 5% for validation
train, valid = train_test_split(
    train,
    test_size=.05,
    stratify=train["type"],
)
print(f"{len(train) = }, {len(valid) = }")


def calculate_output_bias(train):
    """ We can get a reasonable guess for the output bias by just assuming the crystal's
     energy is a linear sum over it's element types """
    # This just converts to a count of each element by crystal
    site_counts = (
        train.inputs.progress_apply(lambda x: pd.Series(Counter(x["site"])))
        .reindex(columns=np.arange(max_atomic_num))
        .fillna(0)
    )
    # Linear regression assumes a sum, while we average over sites in the neural network
    # Here, we make the regression target the total energy, not the site-averaged energy
    num_sites = site_counts.sum(1)
    total_energies = train["energyperatom"] * num_sites

    # Do the least-squares regression, and stack on zeros for the mask and unknown
    # tokens
    output_bias = np.linalg.lstsq(site_counts, total_energies, rcond=None)[0]
    return output_bias


def build_dataset(split, batch_size):
    return (
        tf.data.Dataset.from_generator(
            lambda: ((row.inputs, row.energyperatom) for _, row in split.iterrows()),
            output_signature=(
                preprocessor.output_signature,
                tf.TensorSpec((), dtype=tf.float32),
            ),
        )
        .cache()
        .shuffle(buffer_size=len(split))
        .padded_batch(
            batch_size=batch_size,
            padding_values=(
                preprocessor.padding_values,
                tf.constant(np.nan, dtype=tf.float32),
            ),
        )
        .prefetch(tf.data.experimental.AUTOTUNE)
        #.shuffle(buffer_size, reshuffle_each_iteration=True)
        .repeat()
    )


# Calculate an initial guess for the output bias
output_bias = calculate_output_bias(train)

batch_size = 64
train_dataset = build_dataset(train, batch_size=batch_size)
valid_dataset = build_dataset(valid, batch_size=batch_size)


# Keras model
site_class = layers.Input(shape=[None], dtype=tf.int64, name="site")
distances = layers.Input(shape=[None], dtype=tf.float32, name="distance")
connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name="connectivity")
input_tensors = [site_class, distances, connectivity]

embed_dimension = 256
num_messages = 6

atom_state = layers.Embedding(
    max_atomic_num, embed_dimension, name="site_embedding", mask_zero=True
)(site_class)

atom_mean = layers.Embedding(
    max_atomic_num,
    1,
    name="site_mean",
    mask_zero=True,
    embeddings_initializer=tf.keras.initializers.Constant(output_bias),
)(site_class)

rbf_distance = RBFExpansion(
    dimension=128, init_max_distance=7, init_gap=30, trainable=True
)(distances)

bond_state = layers.Dense(embed_dimension)(rbf_distance)

for _ in range(num_messages):
    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
    bond_state = layers.Add()([bond_state, new_bond_state])
    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])
    atom_state = layers.Add()([atom_state, new_atom_state])

# Reduce the atom state vector to a single energy prediction
atom_state = layers.Dense(
    1,
    name="site_energy_offset",
    kernel_initializer=tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=1e-6, seed=None
    ),
)(atom_state)

# Add this 'offset' prediction to the learned mean energy for the given element type
atom_state = layers.Add(name="add_energy_offset")([atom_state, atom_mean])

# Calculate a final mean energy per atom
out = tf.keras.layers.GlobalAveragePooling1D()(atom_state)

model = tf.keras.Model(input_tensors, [out])


# Train the model
STEPS_PER_EPOCH = math.ceil(orig_num_train / batch_size)  # number of training examples
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    1e-4, decay_steps=STEPS_PER_EPOCH * 50, decay_rate=1, staircase=False
)

wd_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    1e-5, decay_steps=STEPS_PER_EPOCH * 50, decay_rate=1, staircase=False
)

optimizer = tfa.optimizers.AdamW(
    learning_rate=lr_schedule, weight_decay=wd_schedule, global_clipnorm=1.0
)

model.compile(loss="mae", optimizer=optimizer)


filepath = out_dir + "/best_model.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, save_best_only=True, verbose=0
)
csv_logger = tf.keras.callbacks.CSVLogger(out_dir + "/log.csv")

if __name__ == "__main__":
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        # make sure each model trains on the same amount data
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=math.ceil(len(valid)/batch_size),
        epochs=100,
        callbacks=[checkpoint, csv_logger],
        verbose=1,
    )

    data = pd.concat([train, valid, test, test_composition])
    data["set"] = "train"
    data.loc[data.index.isin(valid.index), "set"] = "valid"
    data.loc[data.index.isin(test.index), "set"] = "test"
    data.loc[data.index.isin(test_composition.index), "set"] = "test_composition"

    dataset = (
        tf.data.Dataset.from_generator(
            lambda: (row.inputs for _, row in data.iterrows()),
            output_signature=preprocessor.output_signature,
        )
        .padded_batch(batch_size=128, padding_values=preprocessor.padding_values)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    predictions = model.predict(dataset, verbose=1)

    data["energy_predicted"] = predictions
    data.drop("inputs", axis=1).to_csv(
        out_dir + "/predicted_energies.csv.gz", compression="gzip"
    )
