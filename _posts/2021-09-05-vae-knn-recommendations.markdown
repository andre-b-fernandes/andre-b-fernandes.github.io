---
layout: home
title:  "Autoencoders for product recommendations"
date:   2021-06-26
permalink: /autoencoders-product-recommendations/
categories: recommendations machine-learning computer-vision
image: vae-product-recommendations.png
---

In this article, I'm going to write about Autoencoders and how they can be used to create recommender systems for content-based recommendations using product images.

**What are autoencoders?**

Autoencoders are a type of neural network which attempts to compress long-range multidimensional into a low dimensional space and then decompress it into the original dimension number trying to minimize the error between the decompressed image and the original one.

**How can they be used for product recommendations?**

Most products have associated images for people to see while buying online.
Images are represented as arrays of pixel values where all the pixels have the same dimension which depends on the image representation.
For instance, if we have a colored image of 3 RGB channels with 150px width and 150px height, representing it as a *NumPy* array would result in a shape of `(150, 150, 3)`. If we represented this image as a *1-dimensional* vector, flattening it, it would contain `150*150*3 = 67500` elements.
Calculating vector similarities becomes very difficult with this type of dimensionality.

**How can we implement this using Tensorflow?**

```python
class Autoencoder(Model):
    KERNEL_CONV = (3, 3)
    KERNEL_POOL = (2, 2)

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.zero_pad_layer = ZeroPadding2D((1, 1))

        self.enc_first_conv_2d_layer = Conv2D(16, self.KERNEL_CONV, activation="relu", padding="same")
        self.enc_sec_conv_2d_layer = Conv2D(8, self.KERNEL_CONV, activation="relu", padding="same")
        self.enc_third_conv_2d_layer = Conv2D(8, self.KERNEL_CONV, activation="relu", padding="same")

        self.enc_first_max_pool = MaxPooling2D(self.KERNEL_POOL, padding="same")
        self.enc_sec_max_pool = MaxPooling2D(self.KERNEL_POOL, padding="same")
        self.enc_third_max_pool = MaxPooling2D(self.KERNEL_POOL, padding="same")

        self.dec_first_conv_2d_layer = Conv2D(8, self.KERNEL_CONV, activation="relu", padding="same")
        self.dec_second_conv_2d_layer = Conv2D(8, self.KERNEL_CONV, activation="relu", padding="same")
        self.dec_third_conv_2d_layer = Conv2D(16, self.KERNEL_CONV, activation="relu", padding="same")
        self.dec_fourth_conv_2d_layer = Conv2D(3, self.KERNEL_CONV, activation="sigmoid", padding="same")

        self.dec_first_up_samp_layer = UpSampling2D(self.KERNEL_POOL)
        self.dec_sec_up_samp_layer = UpSampling2D(self.KERNEL_POOL)
        self.dec_third_up_samp_layer = UpSampling2D(self.KERNEL_POOL)

        self.cropping_layer = Cropping2D((1, 1))

    def encode(self, x):
        encoded = self.zero_pad_layer(x)
        encoded = self.enc_first_conv_2d_layer(encoded)
        encoded = self.enc_first_max_pool(encoded)
        encoded = self.enc_sec_conv_2d_layer(encoded)
        encoded = self.enc_sec_max_pool(encoded)
        encoded = self.enc_third_conv_2d_layer(encoded)
        encoded = self.enc_third_max_pool(encoded)
        return encoded

    def decode(self, encoded):
        decoded = self.dec_first_conv_2d_layer(encoded)
        decoded = self.dec_first_up_samp_layer(decoded)
        decoded = self.dec_second_conv_2d_layer(decoded)
        decoded = self.dec_sec_up_samp_layer(decoded)
        decoded = self.dec_third_conv_2d_layer(decoded)
        decoded = self.dec_third_up_samp_layer(decoded)
        decoded = self.dec_fourth_conv_2d_layer(decoded)
        decoded = self.cropping_layer(decoded) 
        return decoded

    def call(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
```

**How can we calculate similarities?**

After we fit the neural network we can use the `encode` function to produce low dimensional representations of flattened images.
We can then make use of `sklearn.neighbors.NearestNeighbors` class to compute a neighborhood of product images.

```python
    from numpy import squeeze
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(enc_flat_images)
    indices = knn.kneighbors(enc_flat_image, return_distance = False)
    neighbors = [original_images[i] for i in squeeze(indices)]
```

**Results**

Full code is hosted in **GitHub** [here](https://github.com/Marko50/product-image-similarity).

I've used all the 150x150 3 RGB channel images present in this dataset [here](https://www.kaggle.com/jonathanoheix/product-recommendation-based-on-visual-similarity).

The original image:
![gru](/assets/img/posts/vae-product-recommendations/original.png)

The decoded image:
![gru](/assets/img/posts/vae-product-recommendations/decoded.png)

The neighborhood
![gru](/assets/img/posts/vae-product-recommendations/1_neighbor.png)
![gru](/assets/img/posts/vae-product-recommendations/2_neighbor.png)
![gru](/assets/img/posts/vae-product-recommendations/3_neighbor.png)
![gru](/assets/img/posts/vae-product-recommendations/4_neighbor.png)




