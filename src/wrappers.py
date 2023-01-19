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


# %% Implementation
# ------------------------------------------------------------------------------


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def mean_rel_diff(score1, score2):
    mean_val = np.abs(np.mean(score1) / 2. + np.mean(score2) / 2.)
    diff = np.mean(np.abs(score1 - score2))
    if diff == 0: 
        return diff
    else: 
        return diff / mean_val


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def normalize_img_data(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32) / 255.
    X = np.expand_dims(X, -1)
    return X


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def load_mnist(n_test_data: int=60000):
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

    ## Reshaping
    y_train = keras.utils.to_categorical(y_train, num_classes)[:n_test_data]
    y_test  = keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train), (X_test, y_test)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def simple_accuracy(model, X, y):
    y_pred = model(X)
    y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
    y      = tf.cast(tf.argmax(y, axis=1), tf.int32)
    return tf.reduce_mean(tf.cast(y_pred == y, tf.float32)).numpy()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def random_batch(X, y, batch_size=128):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def non_random_batch(X, y, n, batch_size=128):
    X_batch = X[n*batch_size : (n+1)*batch_size]
    y_batch = y[n*batch_size : (n+1)*batch_size]
    return X_batch, y_batch


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def print_status_bar(iteration, total, metrics=None):
    metrics = " - ".join([
        "{}: {:.4f}".format(m.name, m.result())
        for m in (metrics or [])
    ])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics,
        end=end)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def prospr_loop(        # TODO: Update to new mode with masked networks
    model: keras.Model,
    X_train: np.ndarray, 
    y_train: np.ndarray,
    loss_fn: keras.losses.Loss,
    metrics: list[keras.metrics.Metric],
    optimizer: keras.optimizers.Optimizer=None,
    validation_data: tuple[np.ndarray]=None,
    prune_sg_lr: float=0.01,
    n_epochs: int=1, 
    batch_size: int=128):    

    ## Warning
    if optimizer is not None:
        print("Warning: Specifying a custom optimizer leads to backpropagation through all batches and instabilities. Consider just using vanilla SGD.")
        c = [tf.ones_like(w) for w in model.trainable_weights]
        masked_model = StaticMaskedModel(c, model)

    ## Settings and History
    n_steps = len(X_train) // batch_size
    fit_history = {'loss': [], 'val_loss': []}
    Jvs = [tf.Variable(var) for var in model.get_weights()]

    for epoch in range(0, n_epochs):
        print("Epoch {}/{}".format(epoch+1, n_epochs))
        for step in range(0, n_steps):

            ## Forward Pass
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)


            ## Vanilla SGD Mode:
            # - - - - - - - - -
            if optimizer is None:
                w_updates = []
                with tf.GradientTape() as outer_tape: ## Pruning Step
                    with tf.GradientTape() as inner_tape: # Update Step
                        y_pred = model(X_batch, training=True)
                        main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                        loss = tf.add_n([main_loss] + model.losses)
                    gradients = inner_tape.gradient(loss, model.trainable_weights)
                    for w, g in zip(model.trainable_weights, gradients):
                        w_updates.append(w - prune_sg_lr * g)

                    int_Jv = tf.reduce_sum([tf.reduce_sum(w_ * tf.stop_gradient(v_)) 
                        for w_, v_ in zip(w_updates, Jvs)])
                Jvs = outer_tape.gradient(int_Jv, model.trainable_weights)
                for w, w_update in zip(model.trainable_weights, w_updates):
                    w.assign(w_update) ## Update


            ## Custom Optimizer Mode:
            # - - - - - - - - - - - -
            if optimizer is not None:
                with tf.GradientTape(persistent=True) as tape: # Update Step
                    y_pred = masked_model(X_batch, training=True)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss] + model.losses)
                gradients = inner_tape.gradient(loss, model.trainable_weights)
                    


            ## Diagnosis
            for metric in metrics:
                metric(y_batch, y_pred)

            ## Validation Loss
            if validation_data is not None:
                val_loss = loss_fn(
                    validation_data[1], 
                    model(validation_data[0], training=False))

            ## Fit History
            fit_history['loss'].append(main_loss.numpy())
            if validation_data is not None:
                fit_history['val_loss'].append(val_loss.numpy())

            ## Diagnosis
            print_status_bar((step+1) * batch_size, len(y_train), metrics)
        print_status_bar(len(y_train), len(y_train), metrics)
        for metric in metrics:
            metric.reset_states()

        return Jvs, fit_history


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def compute_prospr_scores(
    Jvs: list[np.ndarray],
    model: MaskedModel,
    X_train: np.ndarray, 
    y_train: np.ndarray,
    loss_fn: keras.losses.Loss,
    prune_sg_lr: float=0.1):

    ## Final Prune Factor (maybe switch X_train, y_train to X_batch, y_batch ~ last batch)
    with tf.GradientTape() as tape:
        y_pred = model(X_train, training=True)
        main_loss = tf.reduce_mean(loss_fn(y_train, y_pred))
        total_loss = tf.add_n([main_loss] + model.losses)
    final_prune_gradients = tape.gradient(total_loss, model.proper_weights)
    for w, g in zip(model.proper_weights, final_prune_gradients):
        w.assign(w - prune_sg_lr * g) # Final Update step

    meta_gradients = []
    for Jv, g_prune in zip(Jvs, final_prune_gradients):
        meta_gradients.append(Jv * g_prune)

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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def generate_pruning_maskDEPR( ## Total pruning (not layer-wise) DEPRECATED
    scores,
    keep_top: float=0.2):

    flattened_scores = np.concatenate([
        score.numpy().flatten() for score in scores
    ])
    quantile = np.quantile(flattened_scores, q=1-keep_top)

    masks = []
    for score in scores:
        mask = tf.cast(score > quantile, dtype=tf.int16)
        masks.append(mask.numpy())
    
    return masks

def generate_pruning_mask( ## Layer-wise pruning
    scores,
    sparsity: float=0.8, 
    dtype=tf.float32):

    masks = []
    for score in scores:
        score_flat = score.numpy().flatten()
        quantile = np.quantile(
            score_flat, 
            q=np.clip(sparsity, a_min=0., a_max=1.))
        mask = tf.cast(score > quantile, dtype=dtype)
        masks.append(mask)
    
    return masks


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def generate_random_pruning_mask( ## Layer-wise RANDOM pruning
    weight_shapes: list[tf.TensorShape], 
    sparsity: float=0.8):

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
        mask = np.reshape(mask, shp)
        masks.append(tf.constant(mask, dtype=tf.float32))
    
    return masks


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def visualize_prune_history(fit_history: dict):
    steps_arr = np.arange(len(fit_history["loss"]))
    steps_arr_val = steps_arr + 1

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title("Loss vs. Prune-Step", fontsize=22)
    ax.plot(steps_arr,     fit_history['loss'],       "r-",  label="training-data") 
    ax.plot(steps_arr_val, fit_history['val_loss'],   "r--", label="validation-data") 
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss - Categorical Crossentropy")
    leg = ax.legend(frameon=True, fontsize=18)
    return fig


# ------------------------------------------------------------------------------




# %% Main (Test Case)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running `wrappers` test case.")
    print("-----------------------------")

# ------------------------------------------------------------------------------