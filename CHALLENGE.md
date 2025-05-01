# Challenge Faced: Input Shape Mismatch After Preprocessing
While building a binary classification model using Keras, I encountered the following error during model training:

```python
ValueError: Input 0 of layer "dense_3" is incompatible with the layer: 
expected axis -1 of input shape to have value 33, but received input with shape (None, 34)
```

**Cause:**
This occurred because I hardcoded the input shape as `[33]` in the first layer of my model:
```python
layers.Dense(4, activation='relu', input_shape=[33])
```
However, after preprocessing (which included one-hot encoding), the actual number of input features increased to 34. This mismatch caused the model to reject the input during training.

**Solution:**
To resolve the issue, I replaced the hardcoded value with a dynamic one that adapts to the real shape of the training data after preprocessing:
```python
input_shape = [X_train.shape[1]]  # Automatically detects correct feature count
```
This ensured that the model was always compatible with the processed input data, regardless of any feature transformation applied.
