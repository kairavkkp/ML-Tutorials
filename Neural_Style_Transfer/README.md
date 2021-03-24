# Neural-Style-Transfer


<div align="center">
    <img src="https://firebasestorage.googleapis.com/v0/b/merge-my-pdf.appspot.com/o/neural-style-transfer.png?alt=media&token=cd8da818-5ef9-4c00-86d9-affc541c37c9">
</div>

Followed Official Tutorial from Tensorflow: [Link](https://www.tensorflow.org/tutorials/generative/style_transfer)

Contents:
- `Neural_Style_Transfer.ipynb`

Steps:
- Prepare two Images:
    - Content Image
        - The resultant image
    - Style Image
        - The style image that serves as the base for transfer.
- Preprocessing Data
- Experimentation with Pretrained Model on Tensorflow HUB.
    - [Fast Neural Style Transfer](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)
- Manual Creation of Gram Matrix
$$G_{cd}^{l} = \dfrac{\Sigma_{ij} F_{ijc}^{l}(x) F_{ijd}^{l}(x)}{IJ}$$
- Take VGG19 as the base model.
    - Assign Layers
        - For Style (Generally Top Layers)
        - For Content (Generally Bottom Layers)
- Prepare `StyleContentModel` which returns a gram matrix given an image.
- Prepare Functions for Gradient Descend
    - Get Content and Style Targets
    - Optimizers
    - Loss

- Prepare Train Step function.
- Remove High Frequency artifacts from the image by regularizing terms.
    - By calculating `total_variation_loss`.
- Re-running optimizations and Re-training.

Images Used:
- [ContentImage1](https://firebasestorage.googleapis.com/v0/b/merge-my-pdf.appspot.com/o/mountain_house.jpeg?alt=media&token=c072059d-04b2-4249-9cbc-b2efd9ff98ee)
- This image was from `thispersondoesnotexist.com` : [ContentImage2](https://firebasestorage.googleapis.com/v0/b/merge-my-pdf.appspot.com/o/person.jpeg?alt=media&token=75260f18-1dd8-4236-9a15-cd68c8fb36e2)
- [StyleImage1](https://firebasestorage.googleapis.com/v0/b/merge-my-pdf.appspot.com/o/finger-1.jpg?alt=media&token=d6dca7d4-ad4e-4855-811e-9fc9f638b3ce)
- [StyleImage2](https://firebasestorage.googleapis.com/v0/b/merge-my-pdf.appspot.com/o/shading.jpg?alt=media&token=fc3d1198-8750-44eb-bc01-dd4479dcb00a)
- [StyleImage3](https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg)
