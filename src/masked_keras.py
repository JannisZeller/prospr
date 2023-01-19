##### masked_keras #####
# - - - - - - - - - - -
# Source file for masked keras objects (layers and model)
# 
# To run the test case, execute something like 
#   `python -m src.masked_keras` 
# in the terminal.
# ------------------------------------------------------------------------------



# %% Imports
# ------------------------------------------------------------------------------

from typing import Callable
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Conv2D, Dense

# ------------------------------------------------------------------------------




# %% Implementation
# ------------------------------------------------------------------------------


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class MaskedModel(keras.Model):


    def get_masks(self):
        """Returning the masks detached.
        """
        return [w.numpy() for w in self.masks]


    def get_proper_weights(self):
        """Retrieving only the 'proper' kernel-weights.
        """
        return [w.numpy() for w in self.proper_weights]


    def get_weights(self):
        """Retrieving all weights (masks & weights).
        """
        return [w.numpy() for w in self.weights]

    
    def reset(self, model):
        """Resetting the model to the initial weights and masks.
        """
        model.set_weights(self.w_init)
        self = self.__init__(self.masks_init, model)






# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class StaticMaskedModel(MaskedModel):
    def __init__(self, 
        masks: list[tf.Tensor], 
        kernel: keras.Model):
        """ 
        Creates a masked version of the `kernel`-model. The masks are NOT 
        trackable for `tf.GradientTape` and alike because weight masking is
        detached from the inputs (see `call`-method). 

        Parameters
        ----------
        masks : list[tf.Tensor]
            List of Tensors containing the masks for all layers of the model. 
            Can be of any dtype, that is convertible to floats. 
            Typically only containing 0 or 1.
        kernel
            The model whichs parameters should be masked.
        """
        super().__init__()

        ## For Reset
        self.w_init = kernel.get_weights()
        if isinstance(masks[0], tf.Tensor):
            self.masks_init  = [mask.numpy() for mask in masks]
        elif isinstance(masks[0], np.ndarray):
            self.masks_init  = [mask for mask in masks]
        else:
            raise(ValueError, "No suitable type of masks provided.")

        ## Internals
        self.kernel = kernel
        self.proper_weights = kernel.trainable_weights
        self.proper_dtypes = [w.dtype for w in self.proper_weights]
        self.proper_names  = [w.name[:-2]  for w in self.proper_weights]
        self.masks  = [
            tf.constant(tf.cast(mask, dtype=w.dtype))
            for mask, w in zip(masks, kernel.trainable_weights)]



        ## Everything above to base class
        # ░█──░█ ░█▀▀▀█ ░█▀▀█ ░█─▄▀ ▀█▀ ░█▄─░█ ░█▀▀█     ░█─░█ ░█▀▀▀ ░█▀▀█ ░█▀▀▀
        # ░█░█░█ ░█──░█ ░█▄▄▀ ░█▀▄─ ░█─ ░█░█░█ ░█─▄▄     ░█▀▀█ ░█▀▀▀ ░█▄▄▀ ░█▀▀▀
        # ░█▄▀▄█ ░█▄▄▄█ ░█─░█ ░█─░█ ▄█▄ ░█──▀█ ░█▄▄█     ░█─░█ ░█▄▄▄ ░█─░█ ░█▄▄▄

    def call(self, inputs):
        """Forward pass, applying mask.
        """
        for w, mask in zip(self.kernel.trainable_weights, self.masks):
            # w.assign( w * mask )
            w.assign(w * mask)
        return self.kernel(inputs)




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - -
class TrackableMaskedModel(MaskedModel):
    def __init__(self, 
    masks: list[tf.Tensor], 
    kernel: keras.Model,
    # call_overrides: dict() # Future: TODO: Pass call_overrides as a dict(clsname: forwad_call_fun(layer, input))
    ):
        """ 
        Creates a masked version of the `kernel`-model. The masks ARE 
        trackable for `tf.GradientTape` and alike. The drawback is, that the
        forward pass has to be overwritten for each and every layer such that
        the masks directly affect the inputs.

        Parameters
        ----------
        masks : list[tf.Tensor]
            List of Tensors containing the masks for all layers of the model. 
            Can be of any dtype, that is convertible to floats. 
            Typically only containing 0 or 1.
        kernel
            The model whichs parameters should be masked.
        """
        super().__init__()

        ## For Reset
        self.w_init = kernel.get_weights()
        if isinstance(masks[0], tf.Tensor):
            self.masks_init  = [mask.numpy() for mask in masks]
        elif isinstance(masks[0], np.ndarray):
            self.masks_init  = [mask for mask in masks]
        else:
            raise(ValueError, "No suitable type of masks provided.")

        ## Internals
        self.kernel = kernel
        self.proper_weights = kernel.trainable_weights
        self.proper_dtypes = [w.dtype for w in self.proper_weights]
        self.proper_names  = [w.name[:-2]  for w in self.proper_weights]
        self.masks  = [
            tf.Variable(
                tf.cast(mask, dtype=w.dtype), 
                trainable=True,
                name=w.name[:-2]+"/mask")
            for mask, w in zip(masks, kernel.trainable_weights)]



        ## Everything above to base class
        # ░█──░█ ░█▀▀▀█ ░█▀▀█ ░█─▄▀ ▀█▀ ░█▄─░█ ░█▀▀█     ░█─░█ ░█▀▀▀ ░█▀▀█ ░█▀▀▀
        # ░█░█░█ ░█──░█ ░█▄▄▀ ░█▀▄─ ░█─ ░█░█░█ ░█─▄▄     ░█▀▀█ ░█▀▀▀ ░█▄▄▀ ░█▀▀▀
        # ░█▄▀▄█ ░█▄▄▄█ ░█─░█ ░█─░█ ▄█▄ ░█──▀█ ░█▄▄█     ░█─░█ ░█▄▄▄ ░█─░█ ░█▄▄▄



        ## Overriding Forward Passes
        #  If Biases are not used this has to be changed!
        def masked_dense_forward(self, inputs):
            return self.activation(
                tf.add(
                    tf.matmul(
                        inputs,
                        self.weights[0] * self.mask[0]),
                    self.weights[1] * self.mask[1]))

        def masked_conv2D_forward(self, inputs):
            return self.activation(
                tf.add(
                    tf.nn.conv2d(
                        input=inputs, 
                        filters=self.kernel * self.mask[0], 
                        strides=self.strides, 
                        padding=self.padding.upper()),
                    self.bias * self.mask[1]))

        mask_idx = 0
        for layer in self.kernel.layers:

            if isinstance(layer, Dense):
                print("Masking", layer, sep=" ")
                mask_idx_up = mask_idx + len(layer.weights)
                layer.mask = self.masks[mask_idx : mask_idx_up]
                layer.call = masked_dense_forward.__get__(layer, Dense)
                mask_idx = mask_idx_up

            if isinstance(layer, Conv2D):
                print("Masking", layer, sep=" ")
                mask_idx_up = mask_idx + len(layer.weights)
                layer.mask = self.masks[mask_idx : mask_idx_up]
                layer.call = masked_conv2D_forward.__get__(layer, Dense)
                mask_idx = mask_idx_up


    def call(self, inputs):
        """Forward pass, mask is applied at each layer internally.
        """
        return self.kernel(inputs)


# ------------------------------------------------------------------------------




# %% Main (Test Case)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running `masked_keras` test case.")
    print("---------------------------------")

# ------------------------------------------------------------------------------