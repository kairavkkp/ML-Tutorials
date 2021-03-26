# Autoencoder

Autoencoder is a neural network that learns to encode the given inputs and tries to recreate the input as the output, minimizing the reconstruction error. Eventually it tries to compress the input data.

## Types of Autoencoder Used for:
1) Basic Autoencoder

    In this example the network is supposed to recreate the input image, by down-sampling them into a latent space with `64` dimension and expected to up-sample the same into the original image.

In the below eg., Images were passed to the network, being converted (encoded) into latent space and decoded back to original.

<div>
    <img src="images/basic_encoder.png" align="center">
</div>
<br>

2) De-noise Autoencoder


    In this example, random gaussian noise is added to the image by a `noise_factor`. The network is expected to remove the noise from the images. 

    The training is done by keeping the training data as the noisy images and the training labels are kept as the original images. What this means is that the network is training itself to remove the noise at the time of reconstruction.

<div>
    <img src="images/Denoise.png" align="center">
</div>
<br>

3) Anomaly Detection

    In this example ECG data is being used, which comprises of anomalous data points. The data is properly labelled to help us train the model exactly as we want, i.e., Only on Normal Data. 

    Later on we train the Autoencoder with only normal train data, and tune it accordingly to reconstruct it back. The reconstruction error serves as a threshold to predict if the data is normal or anomalous.

<div>
<table>
    <tr>
        <th><div align="center">Original</div></th>
        <th><div align="center">Anomaly</div></th>
    </tr>
    <tr>
        <td>
            <div>
                <img src="images/ECG.png" align="center">
            </div>
        </td>
        <td>
            <div>
                <img src="images/Anomaly.png" align="center">
            </div>
        </td>
    </tr>
</table>
   
</div>

<br>


