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
    def __init__(self, 
        masks: list[tf.Tensor], 
        kernel: keras.Model):
        """ 
        Creates a masked version of the `kernel`-model. This abstract class is 
        used as a template for the two versions below - one with trainable masks
        and one with fixed masks. 

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
        #  If the kernel-model itself gets stored / copied for reset-purposes
        #  this model gets additional trainable parameters, that do not have any
        #  Purpose and cause error. Therefore just store the weights and 
        #  reconstruct the model structure externally (via a function).
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
        super().__init__(masks, kernel)
        ## Creating Masks
        self.__create_masks(masks, kernel)


    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass, applying mask.
        """
        for w, mask in zip(self.kernel.trainable_weights, self.masks):
            # w.assign( w * mask )
            w.assign(w * mask)
        return self.kernel(inputs)


    def __create_masks(self, 
        masks: list[tf.Tensor], 
        kernel: keras.Model):
        """Creating untrainable Masks
        """
        self.masks  = [
            tf.constant(tf.cast(mask, dtype=w.dtype))
            for mask, w in zip(masks, kernel.trainable_weights)]




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
        super().__init__(masks, kernel)
        ## Creating Masks
        self.__create_masks(masks, kernel)

        ## Overriding Forward Passes
        #  If Biases are not used this has to be changed!
        def masked_dense_forward(self, inputs):
            return self.activation(
                tf.add(
                    tf.matmul(
                        inputs,
                        self.weights[0] * self.masks[0]),
                    self.weights[1] * self.masks[1]))

        def masked_conv2D_forward(self, inputs):
            return self.activation(
                tf.add(
                    tf.nn.conv2d(
                        input=inputs, 
                        filters=self.kernel * self.masks[0], 
                        strides=self.strides, 
                        padding=self.padding.upper()),
                    self.bias * self.masks[1]))

        mask_idx = 0
        self.unmasked_forwards = []
        for layer in self.kernel.layers:

            if isinstance(layer, Dense):
                self.unmasked_forwards.append(layer.call)
                mask_idx = self.__override_forward(
                    layer, mask_idx, 
                    masked_dense_forward.__get__(layer, Dense))

            if isinstance(layer, Conv2D):
                self.unmasked_forwards.append(layer.call)
                mask_idx = self.__override_forward(
                    layer, mask_idx, 
                    masked_conv2D_forward.__get__(layer, Conv2D))


    def call(self, inputs) -> tf.Tensor:
        """Forward pass, mask is applied at each layer internally.
        """
        return self.kernel(inputs)

    
    def unmask_forward_passes(self) -> None:
        ## Reset layers forward passes (for ProsPr after first call.)
        mask_idx = 0
        for layer in self.kernel.layers:
            if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                layer.call = self.unmasked_forwards[mask_idx]
                mask_idx +=1


    def __create_masks(self, 
        masks: list[tf.Tensor], 
        kernel: keras.Model) -> None:
        """Creating trainable Masks
        """
        self.masks  = [
            tf.Variable(
                tf.cast(mask, dtype=w.dtype), 
                trainable=True,
                name=w.name[:-2]+"/mask")
            for mask, w in zip(masks, kernel.trainable_weights)]


    def __override_forward(self,
        layer: keras.layers.Layer, 
        mask_idx: int, 
        masked_forwad: Callable) -> int:
        """Abbreviator to override forward passes.

        Returns
        -------
            Index of the next layers first mask.
        """        
        print("Masking", layer, sep=" ")
        mask_idx_up = mask_idx + len(layer.weights)
        layer.masks = self.masks[mask_idx : mask_idx_up]
        layer.call = masked_forwad
        return mask_idx_up


# ------------------------------------------------------------------------------




# %% Main (Test Case)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running `masked_keras` test case.")
    print("---------------------------------")

# ------------------------------------------------------------------------------