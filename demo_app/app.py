import gradio as gr
import tensorflow as tf
from glob import glob
import numpy as np

model_path = "BasicModel"
model = tf.keras.models.load_model(model_path)
labels = [str(i) for i in range(10)]

example_images = glob('sample_images/*.png')

# loading the classes & type casting the encoding indexes
def process_image(image):
    # Convert into tensor
    image = tf.convert_to_tensor(image)

    # Cast the image to tf.float32
    image = tf.cast(image, tf.float32)
    
    # Resize the image to img_resize
    image = tf.image.resize(image, (32,32))
    
    # Normalize the image
    image /= 255.0
    
    # Return the processed image and label
    return image

def predict(image):

  # Pre-procesing the data
  images = process_image(image)

  # Batching
  batched_images = tf.expand_dims(images, axis=0)
  
  prediction = model.predict(batched_images).flatten()
  confidences = {labels[i]: np.round(float(prediction[i]), 3) for i in range(len(labels))}
  return confidences

# creating the component
demo = gr.Interface(fn=predict, 
             inputs=gr.Image(shape=(32, 32)),
             outputs=gr.Label(num_top_classes=len(labels)),
             examples=example_images)
            
# Launching the demo
if __name__ == "__main__":
    demo.launch()
