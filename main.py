import cv2 
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import io

# get grayscale image
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

# erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

# Display the original and processed images
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 3, 1)
# plt.title('Original Image')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.title('Grayscale Image')
# plt.imshow(grayscale(img), cmap='gray')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.title('Sharpened Image')
# plt.imshow(thresholding(grayscale(img)), cmap='gray')
# plt.axis('off')

# plt.show()

uploaded_file = st.sidebar.file_uploader("Bir resim dosyası seçin")

if uploaded_file:
    img = Image.open(io.BytesIO(uploaded_file.getvalue()))
else:
    img_path = "images/image1.jpeg"
    img = cv2.imread(img_path)

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Adding custom options
custom_config = r"--oem 3 --psm 6 l"

text = pytesseract.image_to_string(img, lang='eng+tur', config=custom_config)
# print(text)

# Title of the app
st.title("Tera Text Recognation")
st.subheader("Örnek Resim")

# Display an image
# image = Image.open(img_path)  # Replace with your image file path
st.image(img, caption="Sample image", use_column_width="auto")

st.subheader("Tanımlanan metin")
st.text(text)
