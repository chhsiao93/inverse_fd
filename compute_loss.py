from imageio.v3 import imread
import numpy as np


def compute_image_loss(image, image_ref):
    return (np.sqrt(np.mean((image - image_ref)**2)))


n_view = 6
images_ref = [(imread(f'data/image_ref{i:02d}.png'))/255.0 for i in range(n_view)]
images_ref = np.array(images_ref)
images = [(imread(f'data/image_{i:02d}.png'))/255.0 for i in range(n_view)]
images = np.array(images)
print(images.max(),images.min(),images_ref.max(),images_ref.min())
for image, image_ref in zip(images, images_ref):
    loss = compute_image_loss(image, image_ref)
    print(loss)