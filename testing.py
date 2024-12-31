import tensorflow as tf
import tf2onnx
import os
import keras

# Enable unsafe deserialization
keras.config.enable_unsafe_deserialization()

# Load the Keras model
keras_model = tf.keras.models.load_model("Models/03_handwriting_recognition/202412311722/model.keras", compile=False)

# Convert the model to ONNX format
onnx_model_path = "Models/03_handwriting_recognition/202412311722/model.onnx"
spec = (tf.TensorSpec((None, *keras_model.input_shape[1:]), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=13)

# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())