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
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

# ------------------------------------------------------------------------------




# %% Implementation
# ------------------------------------------------------------------------------

class MaskedLayer(layers.Layer):
    """Class for wrapping keras.layers with a mask.
    """
    def __init__(self, masks: list[tf.Tensor], kernel: layers.Layer):
        """
        Parameters
        ----------
        masks : list[tf.Tensor]
            List of Tensors containing the masks for each weight-object of the
            layer. Can be of any dtype, that is convertible to floats. 
            Typically only containing 0 or 1.
        kernel
            The layer whichs parameters should be masked
        """
        super().__init__()
        self.kernel = kernel
        self.masks = masks
    def build(self, input_shape):
        """Building the kernel layer.
        This can be used when the layers themself do not yet exist.
        """
        self.kernel.build(input_shape)
    def call(self, inputs):
        """Forward pass, applying mask.
        """
        for w, mask in zip(self.kernel.weights, self.masks):
            w.assign( w * tf.cast(mask, dtype=w.dtype) )
        return self.kernel(inputs)


class MaskedModel(keras.Model):
    def __init__(self, masks: list[tf.Tensor], kernel: keras.Model):
        """ 
        Parameters
        ----------
        masks : list[tf.Tensor]
            List of Tensors containing the masks for all layers of the model. 
            Can be of any dtype, that is convertible to floats. 
            Typically only containing 0 or 1.
        kernel
            The layer whichs parameters should be masked. 
            IMPORTANT: Has to be built already!
        """
        super().__init__()
        self.kernel = kernel
        self.masks = masks
    def call(self, inputs):
        """Forward pass, applying mask.
        """
        for w, mask in zip(self.kernel.trainable_weights, self.masks):
            w.assign( w * tf.cast(mask, dtype=w.dtype) )
        return self.kernel(inputs)

# ------------------------------------------------------------------------------




# %% Main (Test Case)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running `masked_keras` test case.")
    print("---------------------------------")

# ------------------------------------------------------------------------------