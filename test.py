import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
from config import MODEL_PATH, IMAGE_PATH, CHECKPOINT_PATH

def preprocess_image_for_model(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)   
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [256, 256])
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0  
    return tf.expand_dims(img, 0)

def show_results(input_tensor, output_tensor):
    input_np = (tf.squeeze(input_tensor, 0) + 1.0) / 2.0
    output_np = (tf.squeeze(output_tensor, 0) + 1.0) / 2.0

    input_np = tf.clip_by_value(input_np, 0.0, 1.0).numpy()
    output_np = tf.clip_by_value(output_np, 0.0, 1.0).numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(input_np)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Ghibli-styled Output")
    plt.imshow(output_np)
    plt.axis('off')
    plt.show()

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

if not os.path.exists(MODEL_PATH) and not os.path.exists(CHECKPOINT_PATH + ".index"):
    raise FileNotFoundError(f"Neither .h5 model nor checkpoint found. Check MODEL_PATH or CHECKPOINT_PATH.\n"
                            f"MODEL_PATH exists: {os.path.exists(MODEL_PATH)}\n"
                            f"CHECKPOINT .index exists: {os.path.exists(CHECKPOINT_PATH + '.index')}")

generator = None
last_error = None

try:
    InstanceNormalization = pix2pix.InstanceNormalization
    print("Trying tf.keras.models.load_model with InstanceNormalization from tensorflow_examples...")
    generator = tf.keras.models.load_model(MODEL_PATH,
                                          custom_objects={"InstanceNormalization": InstanceNormalization},
                                          compile=False)
    print("Loaded model from .h5 using custom_objects.")
except Exception as e:
    last_error = e
    print("load_model with custom_objects failed:", str(e))

if generator is None:
    try:
        print("Reconstructing generator architecture with pix2pix.unet_generator and trying load_weights()...")
        generator = pix2pix.unet_generator(3, norm_type='instancenorm')
        generator.load_weights(MODEL_PATH)
        print("Weights loaded into reconstructed generator using load_weights().")
    except Exception as e:
        last_error = e
        generator = None
        print("Reconstruct + load_weights failed:", str(e))

if generator is None:
    try:
        print("Attempting to restore generator from checkpoint:", CHECKPOINT_PATH)
        generator = pix2pix.unet_generator(3, norm_type='instancenorm')
        ckpt = tf.train.Checkpoint(generator_g=generator)
        ckpt.restore(CHECKPOINT_PATH).expect_partial()
        print("Restored generator from checkpoint.")
    except Exception as e:
        last_error = e
        generator = None
        print("Checkpoint restore failed:", str(e))

if generator is None:
    raise RuntimeError("Failed to load generator. Last error:\n" + repr(last_error))

input_img = preprocess_image_for_model(IMAGE_PATH)
pred = generator(input_img, training=False)

show_results(input_img, pred)
