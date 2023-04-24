##### wrappers #####
# - - - - - - - - - - -
# Source file for bare wrapper functions. Source: 
# https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/
# 
# To run the test case, execute something like 
#   `python -m src.wrappers` 
# in the terminal.
# ------------------------------------------------------------------------------



# %% Imports
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from .masked_keras import MaskedModel, StaticMaskedModel, TrackableMaskedModel

# ------------------------------------------------------------------------------


# %% Utils
# ------------------------------------------------------------------------------

def mean_rel_diff(x1, x2):
    """Mean-Relative Difference between to tensors of arbitrary shape
    """
    ## Calculating the mean magnitude for each element
    means = ( np.abs(x1) + np.abs(x2) ) / 2.
    ## Replacing nans with infs (means=0 where diffs=0)
    means[means==0] = np.inf
    ## Calculating the absolute differences 
    diffs = np.abs(x1 - x2)
    ## Calculaing relative (to mean magnitude) absolute differences
    rel_diffs = diffs / means
    
    return np.mean(rel_diffs)


def normalize_img_data(X: np.ndarray) -> np.ndarray:
    """Img-Preprocessing
    """
    X = X.astype(np.float32) / 255.
    X = np.expand_dims(X, -1)
    return X


def load_mnist(n_test_data: int=60000):
    """MNist loading and preprocessing
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    assert X_train.shape == (60000, 28, 28)
    assert X_test.shape  == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape  == (10000,)
    num_classes = 10
    assert set(np.unique(y_train)) == set(np.arange(num_classes))

    if n_test_data > 60000:
        print("Requested more training data than possible. Getting maximum 60k.")
        n_test_data = 60000

    ## Preprocessing
    X_train = normalize_img_data(X_train)[:n_test_data]
    X_test  = normalize_img_data(X_test)

    ## Shuffeling
    shuffler = np.random.permutation(len(y_train))
    X_train = X_train[shuffler]
    y_train = y_train[shuffler]

    ## Reshaping
    y_train = keras.utils.to_categorical(y_train, num_classes)[:n_test_data]
    y_test  = keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train), (X_test, y_test)


def model_accuracy(model, X, y):
    """Accuracy Macro
    """
    y_pred = model(X)
    y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
    y      = tf.cast(tf.argmax(y, axis=1), tf.int32)
    return tf.reduce_mean(tf.cast(y_pred == y, tf.float32)).numpy()


def random_batch(X, y, batch_size=128):
    """Generates random batches from data
    """
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


def non_random_batch(X, y, n, batch_size=128):
    """Generates deterministic batches from data (for reproducability).
    """
    n_max = X.shape[0] // batch_size
    n = n%n_max
    X_batch = X[n*batch_size : (n+1)*batch_size]
    y_batch = y[n*batch_size : (n+1)*batch_size]
    return X_batch, y_batch




## ProsPr Scores: 
#  These would be the last necessairy steps to calculate the scores if the 
#  calculation of meta gradients and / or streaming product would work.
# ------------------------------------------------------------------------------

def compute_prospr_scores(
    Jvps: list[np.ndarray],
    model: MaskedModel,
    X_train: np.ndarray, 
    y_train: np.ndarray,
    loss_fn: keras.losses.Loss,
    prune_sg_lr: float=0.1):

    ## Final Prune Factor (maybe switch X_train, y_train to X_batch, y_batch ~ last batch) 
    with tf.GradientTape() as tape:
        y_pred = model(X_train, training=True)
        loss = loss_fn(y_train, y_pred)
    final_prune_gradients = tape.gradient(loss, model.proper_weights)
    for w, g in zip(model.proper_weights, final_prune_gradients):
        w.assign(w - prune_sg_lr * g) # Final Update step

    meta_gradients = []
    for Jvp, g_prune in zip(Jvps, final_prune_gradients):
        meta_gradients.append(Jvp * g_prune)

    ## Generate Saliency Scores
    denominator = 0
    for meta_g in meta_gradients:
        denominator += tf.reduce_sum(meta_g)
    scores = []
    for meta_g in meta_gradients:
        scores.append(tf.abs(meta_g / denominator))

    return scores


def compute_prospr_scores_unmasked_model(
    Jvps: list[np.ndarray],
    model: keras.Model,
    X_train: np.ndarray, 
    y_train: np.ndarray,
    loss_fn: keras.losses.Loss,
    prune_sg_lr: float=0.1):

    ## Final Prune Factor (maybe switch X_train, y_train to X_batch, y_batch ~ last batch) 
    with tf.GradientTape() as tape:
        y_pred = model(X_train, training=True)
        loss = loss_fn(y_train, y_pred)
    final_prune_gradients = tape.gradient(loss, model.trainable_weights)
    for w, g in zip(model.trainable_weights, final_prune_gradients):
        w.assign(w - prune_sg_lr * g) # Final Update step

    meta_gradients = []
    for Jvp, g_prune in zip(Jvps, final_prune_gradients):
        meta_gradients.append(Jvp * g_prune)

    ## Generate Saliency Scores
    denominator = 0
    for meta_g in meta_gradients:
        denominator += tf.reduce_sum(meta_g)
    scores = []
    for meta_g in meta_gradients:
        scores.append(tf.abs(meta_g / denominator))

    return scores


def compute_prospr_scores_from_meta_g(meta_gradients: list[np.ndarray]):

    ## Generate Saliency Scores
    denominator = 0
    for meta_g in meta_gradients:
        denominator += tf.reduce_sum(meta_g)
    scores = []
    for meta_g in meta_gradients:
        scores.append(tf.abs(meta_g / denominator))

    return scores




## Pruning Mask Generators
# ------------------------------------------------------------------------------

def generate_total_pruning_mask(
    scores,
    keep_top: float=0.2):
    """Generates total (not layer-wise) pruning masks from scores. 
    """
    if isinstance(scores[0], tf.Tensor):
        flattened_scores = np.concatenate([
            score.numpy().flatten() for score in scores
        ])
    if isinstance(scores[0], np.ndarray):
        flattened_scores = np.concatenate([
            score.flatten() for score in scores
        ])
    quantile = np.quantile(flattened_scores, q=1-keep_top)

    masks = []
    for score in scores:
        mask = tf.cast(score > quantile, dtype=tf.int16)
        masks.append(mask.numpy())
    
    return masks


def generate_pruning_mask(
    scores,
    sparsity: float=0.8, 
    dtype=tf.float32):
    """Generates layer-wise pruning masks from scores.
    """
    if isinstance(scores[0], tf.Tensor):
        scores = [score.numpy() for score in scores]
    masks = []
    for score in scores:
        score_flat = score.flatten()
        quantile = np.quantile(
            score_flat, 
            q=np.clip(sparsity, a_min=0., a_max=1.))
        mask = tf.cast(score > quantile, dtype=dtype).numpy() 
        masks.append(mask)
    
    return masks


def generate_random_pruning_mask( ## Layer-wise RANDOM pruning
    weight_shapes: list[tf.TensorShape], 
    sparsity: float=0.8):
    """Generates layer-wise random pruning masks.
    """
    masks = []
    for shp in weight_shapes:
        n_params_weights = shp.num_elements()
        p0 = np.clip(sparsity, a_min=0., a_max=1.)
        p1 = np.clip(1. - p0, a_min=0., a_max=1.)
        n0 = int(np.round(p0 * n_params_weights, 0))
        n1 = int(np.round(p1 * n_params_weights, 0))

        choices = n0*[0] + n1*[1]

        if n0 + n1 > n_params_weights:
            drop_choice = np.random.choice(2, size=1, p=[p1, p0])
            choices.remove(drop_choice)

        if n0 + n1 < n_params_weights:
            additional_choice = np.random.choice(2, size=1, p=[p0, p1])
            choices.append(additional_choice)

        assert len(choices) == n_params_weights, "Number of choices not correct."

        if np.all(np.array(choices) == 0):
            choices[0] = 1
        
        mask = np.random.permutation(choices)
        masks.append(np.reshape(mask, shp))
    
    return masks

# ------------------------------------------------------------------------------