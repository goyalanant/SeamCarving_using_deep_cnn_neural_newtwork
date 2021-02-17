import numpy as np
import cv2
from time import sleep
import streamlit as st
from PIL import Image
import time
carves_per_second = 18
dt = int(1000 / carves_per_second)

def gradient_magnitude(image):
    sobelx = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3))
    sobely = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3))

    grad_mag = np.clip(sobelx + sobely, 0, 255).astype(np.uint8)
    return grad_mag

def get_energy_DP(energy):
    h, w = energy.shape
    
    DP = np.zeros_like(energy, dtype = np.uint32)
    DP[h - 1] = energy[h - 1].astype(DP.dtype)

    for x in range(h - 2, -1, -1):
        for y in range(w):
            _min = DP[x + 1][y]
            if y - 1 >= 0:
                _min = min(_min, DP[x + 1][y - 1])
            if y + 1 < w:
                _min = min(_min, DP[x + 1][y + 1])

            DP[x][y] = energy[x][y] + _min

    return DP

def find_seam(M):
    h, w = M.shape
    seam = [np.argmin(M[0])]
    
    for x in range(1, h):
        y = seam[-1]
        y_min = y
        
        if y - 1 >= 0 and M[x][y - 1] < M[x][y_min]:
            y_min = y - 1
        if y + 1 <  w and M[x][y + 1] < M[x][y_min]:
            y_min = y + 1

        seam.append(y_min)

    return seam
        
def carve_seam(image, seam):
    shape = image.shape
    shape = (shape[0], shape[1] - 1) + shape[2:]
    
    carved = np.zeros(shape, dtype = image.dtype)
    for x, y in enumerate(seam):
        row = np.delete(image[x], y, axis = 0)
        carved[x] = row

    return carved

def color_seam(image, seam, color = [255, 0, 0]):
    for x, y in enumerate(seam):
        image[x][y] = color

    return image


def main():
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
            i = Image.open(uploaded_file)
            image = np.array(i)
            st.write("<h3 style='color: black; font-size: 5;'>ORIGINAL IMAGE</h3>", unsafe_allow_html=True)
            st.image(image)
            t=st.empty()
            imagelocation = st.empty()
            
    if st.button("Run"):
        Start = time.perf_counter()
        h, w = image.shape[:2]
        if len(image.shape) == 3: # If we have a color image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
        else:
            gray = image

        grad_mag = gradient_magnitude(gray)

        grad_mag_3ch = np.zeros((h, w, 3), dtype = np.uint8)
        grad_mag_3ch[:, :, 0] = grad_mag
        grad_mag_3ch[:, :, 1] = grad_mag
        grad_mag_3ch[:, :, 2] = grad_mag
            
        for _ in range(w // 3):
            t.write("<h3 style='color: black; font-size: 5;'>PROCESSING...</h3>", unsafe_allow_html=True)
            DP = get_energy_DP(grad_mag)
            seam = find_seam(DP)

            image = carve_seam(image, seam)

            grad_mag_3ch = color_seam(grad_mag_3ch, seam)
            imagelocation.image(grad_mag_3ch)

            grad_mag = carve_seam(grad_mag, seam)
            grad_mag_3ch = carve_seam(grad_mag_3ch, seam)

        t.empty() 
        st.write("<h3 style='color: black; font-size: 5;'>FINAL OUTPUT</h3>", unsafe_allow_html=True)
        st.image(image)
        end = time.perf_counter()
        sec = end - Start
        ty_res = time.gmtime(sec)
        res = time.strftime("%H:%M:%S",ty_res)
        st.write(f"<h4 style='color: black; font-size: 5;'>Time taken =  {res} </h4>",unsafe_allow_html=True)
        
