import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def resize_image(gray_img,target_size=(256, 256)):
    gray_img = gray_img.resize(target_size)
    gray_img_array = np.array(gray_img).astype('float32')
    gray_img_array = (gray_img_array - 127.5) / 127.5
    gray_img_array = gray_img_array.reshape(1, 256, 256, 1)
    return gray_img_array



# Load the generator model
g_model = tf.keras.models.load_model('project/g_model')

# Load and preprocess the grayscale image
gray_img = Image.open('project/Test/image0918.jpg').convert('L')
gray_img_array = resize_image(gray_img=gray_img)

# Generate output using the generator model
gen_out = g_model.predict(gray_img_array)

# Revert normalization
gen_out = (gen_out + 1) / 2.0
gen_out = gen_out.reshape(256, 256, 3)

# Display the generated image using matplotlib
plt.imshow(gen_out)
plt.axis('off')  # Hide axis
plt.show()
