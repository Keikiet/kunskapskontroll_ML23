import streamlit as st
import numpy as np
from skimage import io, color, transform, filters, util
import joblib
from joblib import dump
from PIL import Image
import matplotlib.pyplot as plt

# Load the MNIST model
model = joblib.load('/Users/keikietpham/VSCODE/Streamlit/voting_clf_model.joblib')

# Set up the navigation bar
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to", ('Home', 'Upload Image'))

if page == 'Home':
    st.title('MNIST Digit Recognition')
elif page == 'Upload Image':
    st.title('Upload Image')

    # Function to preprocess image for prediction
    def preprocess_image(image):
        if len(image.shape) >= 3:
            # Convert image to grayscale if it has three channels (RGB)
            grayscale_image = color.rgb2gray(image)
        else:
            # Image is already grayscale
            grayscale_image = image

        # Image Inversion
        inverted_image = util.invert(grayscale_image)

        # Resize image to 28x28
        resized_image = transform.resize(inverted_image, (28, 28))
        return resized_image

    # Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=['png','jpg', 'jpeg'])

    # Process uploaded image and display prediction
    if uploaded_file is not None:
        # Read uploaded image using PIL
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert PIL Image to numpy array
        image_array = np.array(image)

        # Apply preprocess_image function
        input_image = preprocess_image(image_array)

        # Perform Otsu's thresholding
        thresh = filters.threshold_otsu(input_image)
        binary_image = input_image > thresh

        # Create axes to display Preprocessed, Histogram, and Thresholded Image
        fig, axes = plt.subplots(ncols=3, figsize=(8, 3))
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 3, 1)
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

        ax[0].imshow(input_image, cmap=plt.cm.gray)
        ax[0].set_title('Preprocessed image')
        ax[0].axis('off')

        ax[1].hist(input_image.ravel(), bins=256)
        ax[1].set_title('Histogram')
        ax[1].axvline(thresh, color='r')

        ax[2].imshow(binary_image, cmap=plt.cm.gray)
        ax[2].set_title('Thresholded')
        ax[2].axis('off')

        # Display the matplotlib figure
        st.pyplot(fig)

        # Button to display prediction
        if st.button('Display Prediction'):
            prediction = model.predict(input_image.flatten().reshape(1, -1))  # Flatten image before reshaping
            digit = prediction[0]
        
            #  Display prediction
            st.write(f"Prediction: {digit}")

        # Button to display raw data
        if st.button('Show Pixel Data'):
            st.write('binary image', binary_image )
            st.write('pixel value image', input_image, )
