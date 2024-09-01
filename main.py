import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

st.image(r"C:\Users\Edwin\Python\bootcamp\FTDSU4_Deep_Learning\computer_vision\images\cnn_model.png")

st.title('Handwritten Digit Recognition')
st.write('The dashboard serves as a straightforward demonstration of a computer vision project that recognizes handwritten digits.')

st.subheader('a) Model Summary')

st.write("The project entails preprocessing images to normalize and resize them for consistency, employing a convolutional neural network (CNN) to extract features from the images, and evaluating the model's accuracy on the test dataset.")
st.image(r"C:\Users\Edwin\Python\bootcamp\FTDSU4_Deep_Learning\computer_vision\images\model_summary.png")

st.write("The accuracy surpasses 95% during the validation of the data.")
st.image(r"C:\Users\Edwin\Python\bootcamp\FTDSU4_Deep_Learning\computer_vision\images\accuracy.png")
st.image(r"C:\Users\Edwin\Python\bootcamp\FTDSU4_Deep_Learning\computer_vision\images\loss.png")

st.subheader('b) Recognition of handwritten digits')
st.write("Please feel free to upload some handwritten digit images to evaluate the model's accuracy and predictions.")

uploaded_image = st.file_uploader("Upload image", type = ["png"])
show_file = st.empty()

# Load the model
cnn_model = tf.keras.models.load_model(r"C:\Users\Edwin\Python\bootcamp\FTDSU4_Deep_Learning\computer_vision\cnn_model.keras")
class_name = [num for num in range(10)]

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')  
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Predict'):
            img_array = preprocess_image(uploaded_image)

            # Make a prediction using the pre-trained model
            result = cnn_model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_name[predicted_class]
            confidence = np.max(result)

            st.success(f'Prediction: {prediction} with confidence: {confidence *100:.1f}%')