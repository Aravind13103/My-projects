import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import plot_model

# Load the trained model
model = load_model('our_model.h5')  # Ensure the path is correct

# Save the model architecture to a file
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Display the saved model architecture image
img = plt.imread('model_architecture.png')
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()
