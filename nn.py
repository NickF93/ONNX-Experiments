import sys, os, gc, time, math

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import tf2onnx
import onnxruntime as ort

from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

from tqdm.auto import tqdm

# Get the current time
current_time = time.time()

# Extract the integer and decimal parts of the current time
integer_part = int(current_time)
decimal_part_as_int = int((current_time - integer_part) * 1e6)  # Convert decimal part to integer

a = max(integer_part, decimal_part_as_int)
b = min(integer_part, decimal_part_as_int)

# Calculate the seed
seed = int(a // b if b != 0 else a)

print(f"Random seed: {seed}")

# Ensure the directory for saving images exists
os.makedirs('/tmp/saveimgs', exist_ok=True)

# Define the path to save the best model
checkpoint_dir = '/tmp/best_model'
checkpoint_path = os.path.join(checkpoint_dir, "best_model.ckpt")

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

gc.collect()

def unet_autoencoder(input_shape=(224, 224, 3), filters=[32, 64, 128, 256, 512], z=128, skipn=3, actf=tf.nn.swish, actf_bttlnk=tf.nn.relu):
    inputs = layers.Input(input_shape)
    skips = []  # List to store the skip connections
    
    # Encoder
    x = inputs
    for i in range(len(filters)):
        x = layers.Conv2D(filters[i], (3, 3), activation=actf, padding='same', use_bias=False)(x)
        x = layers.Conv2D(filters[i], (3, 3), activation=actf, padding='same', use_bias=False)(x)
        if i > len(filters) - (skipn + 1):  # Only add pooling if not at bottleneck
            skips.append(x)  # Save for skip connection
        x = layers.MaxPooling2D((2, 2))(x)
    
    # Bottleneck
    x = layers.Conv2D(z, (3, 3), activation=actf_bttlnk, padding='same', use_bias=False)(x)
    
    # Decoder
    for i in range(len(filters)-1, -1, -1):
        x = layers.Conv2DTranspose(filters[i], (2, 2), strides=(2, 2), padding='same', use_bias=False)(x)
        if i > len(filters) - (skipn + 1):  # Add skip connections as per skipn
            s = skips.pop()
            s = layers.Conv2D(z, (3, 3), activation=actf_bttlnk, padding='same', use_bias=False)(s)
            x = layers.concatenate([x, s])
        x = layers.Conv2D(filters[i], (3, 3), activation=actf, padding='same', use_bias=False)(x)
        x = layers.Conv2D(filters[i], (3, 3), activation=actf, padding='same', use_bias=False)(x)
    
    outputs = layers.Conv2D(input_shape[-1], (1, 1), activation='sigmoid')(x)
    
    model = models.Model(inputs=[inputs], outputs=[outputs], name='u-net')
    return model

unet_model = unet_autoencoder(input_shape=(224, 224, 3), filters=[32, 64, 128, 256, 512])
                              
# Save the model architecture as an image
model_path = '/tmp/model.png'
plot_model(unet_model, to_file=model_path, show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, show_layer_activations=True, show_trainable=True)
print(f"Model architecture saved to {model_path}")

rec_loss = tf.keras.losses.Huber()

def custom_loss(y_true, y_pred):
    # Calculate Huber loss
    huber_loss = rec_loss(y_true, y_pred)
    
    # Calculate SSIM loss
    ssim_loss = tf.reduce_mean(1 - tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    # Combine the losses
    total_loss = 50 * huber_loss + ssim_loss
    
    return total_loss

# Set the path to the dataset
dataset_dir = '/tmp/dataset/train'

# Load the dataset with a validation split
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    label_mode=None,  # Ignore labels, as this is an autoencoder
    image_size=(224, 224),  # Resize images to 224x224
    color_mode="rgb",  # Use RGB images
    batch_size=16,  # Batch size (can be adjusted)
    shuffle=True,  # Shuffle the data
    validation_split=0.1,  # 10% of data for validation
    subset="both",  # Create both training and validation datasets
    seed=seed  # Use the calculated seed
)

def augment_image(image):
    # Random rotation
    angle = tf.random.uniform([], minval=-30, maxval=30, dtype=tf.float32)  # random angle between -30 and 30 degrees
    image = tf.image.rot90(image, k=tf.cast(tf.floor(angle / 90), tf.int32))

    # Random horizontal and vertical flip
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    image = tf.clip_by_value(image, 0.0, 1.0)

    # Random color change (if RGB)
    #if image.shape[-1] == 3:
    #    image = tf.image.random_hue(image, max_delta=0.05)
    #    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    #    image = tf.image.random_brightness(image, max_delta=0.1)
    #    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    image = tf.clip_by_value(image, 0.0, 1.0)
    denoised = tf.clip_by_value(image, 0.0, 1.0)

    # Random Gaussian noise
    beta = tf.random.uniform([], minval=0.0, maxval=10, dtype=tf.float32)
    # Define random max value for normalization between 0.2 and 0.3
    eta = tf.random.uniform([], minval=0.0, maxval=0.3, dtype=tf.float32)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=beta, dtype=tf.float32)
    normalized_noise = (((noise) / (tf.math.reduce_max(noise))) * eta)
    image = tf.clip_by_value(image + normalized_noise, 0.0, 1.0)  # Clip the image to ensure values are in range [0, 1]

    return image, denoised

# Function to normalize the images and apply augmentation (only for training)
def normalize_and_augment(image):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0.0, 1.0]
    image = augment_image(image)  # Apply augmentations
    return image

# Function to normalize images (for validation and testing)
def normalize(image):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0.0, 1.0]
    return image

# Apply normalization to the training and validation datasets
train_dataset = dataset[0].map(lambda x: normalize_and_augment(x))
val_dataset = dataset[1].map(lambda x: normalize(x))

def display_images(batch, title_prefix="Image", max_columns=4):
    # Calculate the number of rows and columns
    batch_size = len(batch)
    columns = min(batch_size, max_columns)  # Limit columns to max_columns
    rows = math.ceil(batch_size / columns)
    
    # Set figure size based on the grid dimensions
    plt.figure(figsize=(columns * 3, rows * 3))
    
    for i in range(batch_size):
        ax = plt.subplot(rows, columns, i + 1)
        plt.imshow(batch[i])
        plt.title(f"{title_prefix} {i+1}")
        plt.axis("off")
    
    plt.show()

# Display a batch of training images
train_batch = next(iter(train_dataset))
display_images(train_batch[0], title_prefix="Training Image (A+N)")
display_images(train_batch[1], title_prefix="Training Image (A)")
train_batch = next(iter(train_dataset))
display_images(train_batch[0], title_prefix="Training Image (A+N)")
display_images(train_batch[1], title_prefix="Training Image (A)")

# Display a batch of validation images
val_batch = next(iter(val_dataset))
display_images(val_batch, title_prefix="Validation Image")

# Instantiate the Adam optimizer with the cosine decay learning rate
initial_learning_rate = 0.0005
first_decay_steps = 107
t_mul = 2.0
m_mul = 0.75
alpha = 0.00001

cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=initial_learning_rate,
    first_decay_steps=first_decay_steps,
    t_mul=t_mul,
    m_mul=m_mul,
    alpha=alpha
)

optimizer = tf.keras.optimizers.Adam(learning_rate=cosine_decay)

# Custom training loop
epochs = 200

# Define training and validation steps
@tf.function(reduce_retracing=True)
def train_step(x):
    x, z = x
    with tf.GradientTape() as tape:
        reconstructed = unet_model(x, training=True)
        loss = custom_loss(z, reconstructed)
    grads = tape.gradient(loss, unet_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, unet_model.trainable_weights))
    return loss

@tf.function(reduce_retracing=True)
def val_step(x):
    reconstructed = unet_model(x, training=False)
    loss = custom_loss(x, reconstructed)
    return loss

# Lists to store the loss values
train_losses = []
val_losses = []

# Initialize a variable to keep track of the best validation loss
best_val_loss = float('inf')

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Training
    train_loss = 0
    with tqdm(train_dataset, desc="Training", unit="batch") as pbar:
        for step, x_batch_train in enumerate(pbar):
            loss = train_step(x_batch_train)
            #train_loss += loss
            train_loss = loss
            
            current_lr = optimizer.learning_rate.numpy()
            # Convert loss to a scalar value to display in tqdm
            loss_value = loss.numpy() if isinstance(loss, tf.Tensor) else loss
            pbar.set_postfix({"Loss": float(loss_value), "LR": current_lr})

    #train_loss /= (step + 1)
    train_losses.append(train_loss)
    print(f"Training loss: {train_loss:.4f}")
    
    # Validation
    val_loss = 0
    with tqdm(val_dataset, desc="Validation", unit="batch") as pbar:
        for step, x_batch_val in enumerate(pbar):
            loss = val_step(x_batch_val)
            #val_loss += loss
            val_loss = loss
            
            # Convert loss to a scalar value to display in tqdm
            loss_value = loss.numpy() if isinstance(loss, tf.Tensor) else loss
            pbar.set_postfix({"Loss": float(loss_value)})

    #val_loss /= (step + 1)
    val_losses.append(val_loss)
    print(f"Validation loss: {val_loss:.4f}")

    # Check if the current validation loss is the best we've seen so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"Validation loss improved to {best_val_loss:.4f}. Saving model...")
        
        # Save the model weights
        unet_model.save_weights(checkpoint_path)

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_losses, label='Training Loss')
plt.plot(range(epochs), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
plt.cla()
plt.clf()
plt.close()
plt.close('all')

# Take one batch of images from the validation set
x_batch_val = next(iter(val_dataset))

# Get the reconstructed images
reconstructed_imgs = unet_model(x_batch_val, training=False)

# Convert tensors to numpy arrays for visualization
x_batch_val_np = x_batch_val.numpy()
reconstructed_imgs_np = reconstructed_imgs.numpy()

# Display original and reconstructed images
n = min(5, x_batch_val.shape[0])  # Display 5 or fewer images
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.squeeze(x_batch_val_np[i]), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # Display reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.squeeze(reconstructed_imgs_np[i]), cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

plt.show()

# Save the images to disk
for i in range(n):
    original_image_path = f'/tmp/saveimgs/original_{i}.png'
    reconstructed_image_path = f'/tmp/saveimgs/reconstructed_{i}.png'
    
    # Save original image
    plt.imsave(original_image_path, np.squeeze(x_batch_val_np[i]), cmap='gray')
    
    # Save reconstructed image
    plt.imsave(reconstructed_image_path, np.squeeze(reconstructed_imgs_np[i]), cmap='gray')
plt.cla()
plt.clf()
plt.close()
plt.close('all')

# Save the model in TensorFlow's SavedModel format
saved_model_path = '/tmp/saved_model'
onnx_model_path = '/tmp/unet_model.onnx'
tf.saved_model.save(unet_model, saved_model_path)
print(f"Model saved to {saved_model_path}")

# Convert the Keras model to ONNX format
model_proto, _ = tf2onnx.convert.from_keras(unet_model, opset=13)

# Save the ONNX model to disk
onnx_model_path = '/tmp/unet_model.onnx'
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"ONNX model saved to {onnx_model_path}")

del unet_model, model_proto

gc.collect()

# Final test: Load ONNX model and perform inference
# Load the ONNX model using ONNX Runtime
onnx_session = ort.InferenceSession(onnx_model_path)

# Prepare a test batch from the dataset
test_batch = next(iter(val_dataset))
test_batch_np = test_batch.numpy()

# Run the inference
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name
onnx_predictions = onnx_session.run([output_name], {input_name: test_batch_np})[0]

# Display the original and ONNX reconstructed images
n = min(5, test_batch_np.shape[0])  # Display 5 or fewer images
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.squeeze(test_batch_np[i]), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # Display ONNX reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.squeeze(onnx_predictions[i]), cmap='gray')
    plt.title("ONNX Reconstructed")
    plt.axis("off")

plt.show()

# Save the ONNX model's output images to disk
for i in range(n):
    onnx_reconstructed_image_path = f'/tmp/saveimgs/onnx_reconstructed_{i}.png'
    
    plt.imsave(onnx_reconstructed_image_path, np.squeeze(onnx_predictions[i]), cmap='gray')

plt.cla()
plt.clf()
plt.close()
plt.close('all')

gc.collect()