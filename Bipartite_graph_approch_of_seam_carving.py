import streamlit as st
import numpy as np
import cv2
from time import sleep
from PIL import Image
import time

import sys
sys.setrecursionlimit(100000)



def energy(image):
    #image = cv2.GaussianBlur(image, (5, 5), 5)
    sobelx = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3))
    sobely = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3))

    grad_mag = np.clip(sobelx + sobely, 0, 255).astype(np.uint8)
    return grad_mag

def get_energy_DP(e):
    h, w = e.shape
    
    DP = np.zeros_like(e, dtype = np.uint32)
    DP[h - 1] = e[h - 1].astype(DP.dtype)

    for x in range(h - 2, -1, -1):
        for y in range(w):
            _min = DP[x + 1][y]
            if y - 1 >= 0:
                _min = min(_min, DP[x + 1][y - 1])
            if y + 1 < w:
                _min = min(_min, DP[x + 1][y + 1])

            DP[x][y] = e[x][y] + _min

    return DP

def w2(i,j,k,A,M):
    if(i < 0 or j < 0):
        return np.NINF
    return (A[k][i] * M[k+1][j])


def cal_f(A,M,row,i,f,conn):
    if(i == -1 or i == -2):
        return 0
    if(f[i] != 0 and conn[i] != -1):
        return conn[i]
    else:
        f1 = cal_f(A,M,row,i-1,f,conn) + w2(i,i,row,A,M)
        f2 = cal_f(A,M,row,i-2,f,conn) + w2(i,i-1,row,A,M) + w2(i-1,i,row,A,M)
        if(f1 > f2):
            f[i] = 1
            conn[i] = f1
        else:
            f[i] = 2
            conn[i] = f2
        return conn[i]

def cal_A_graph(e,M):
    A = np.zeros(e.shape)
    A[0] = e[0]
    n = A.shape[1]-1
    graph = []
    for i in range(0,e.shape[0]-1):
        conn = [-1 for i in range(A.shape[1])]
        f = [0 for i in range(A.shape[1])]
        cal_f(A,M,i,n,f,conn)
        temp = [-1 for i in range(A.shape[1])]
        v = [False for i in range(A.shape[1])]
        for j in range(len(f)-1,-1,-1):
            if(f[j] == 1 and v[j] == False):
                temp[j] = j
                A[i+1][j] = A[i][j] + e[i+1][j]
                v[j] = True
            elif(f[j] == 2 and v[j] == False):
                    temp[j] = j-1
                    temp[j-1] = j
                    A[i+1][j-1] = A[i][j] + e[i+1][j-1]
                    A[i+1][j] = A[i][j-1] + e[i+1][j]
                    v[j-1] = True
                    v[j] = True
        graph.append(temp)
    return A,graph

def sort_by_energy(seams,A):    
    def comparitor(l):
        y = l[-1]
        x = A.shape[0]-1
        return A[x][y]
    seams.sort(key = comparitor)

def find_seam(graph):
    seams = []
    for j in range(len(graph[0])):
        l = [j]
        for i in range(len(graph)):
            j = graph[i][j]
            l.append(j)
        seams.append(l)
    return seams

def carve_seams(seams,percent,image):
    n = (len(seams)*percent)//100
    seams = seams[:n]
    h,w = image.shape[:2]
    ih, iw = image.shape[:2]
    carved = np.empty((ih, iw - n) + image.shape[2:], dtype = image.dtype)
    for x in range(h):
        indexes = [seam[x] for seam in seams]
        row = np.delete(image[x], indexes, axis = 0)
        carved[x] = row

    return carved
def color_seam(image, seam, color = [255, 0, 0]):
    for x, y in enumerate(seam):
        image[x][y] = color

    return image

def display_seams(image,percent,seams):
    n = (len(seams)*percent)//100
    seams = seams[:n]
    for seam in seams:
       image = color_seam(image,seam)

    return image
    

        
def main():
        #img  = image.copy()
    #h,w = image.shape[:2]
    #if len(image.shape) == 3:
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
            i = Image.open(uploaded_file)
            image = np.array(i)
            st.write("<h3 style='color: black; font-size: 5;'>ORIGINAL IMAGE</h3>", unsafe_allow_html=True)
            st.image(image)
    
            
    if st.button("Run"):
        Start = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        e = energy(gray)
        h, w = e.shape[:2]
        grad_mag_3ch = np.zeros((h, w, 3), dtype = np.uint8)
        grad_mag_3ch[:, :, 0] = e
        grad_mag_3ch[:, :, 1] = e
        grad_mag_3ch[:, :, 2] = e
        M = get_energy_DP(e)
        A,graph = cal_A_graph(e,M)
        seams = find_seam(graph)
        sort_by_energy(seams,A)
        n = 20
        grad_image = display_seams(grad_mag_3ch,n,seams)
        st.image(grad_image)
        
        carved = carve_seams(seams,n,image)
        st.write("<h3 style='color: black; font-size: 5;'>FINAL OUTPUT</h3>", unsafe_allow_html=True)
        st.image(carved)
        end = time.perf_counter()
        sec = end - Start
        ty_res = time.gmtime(sec)
        res = time.strftime("%H:%M:%S",ty_res)
        st.write(f"<h4 style='color: black; font-size: 5;'>Time taken = {res} </h4>",unsafe_allow_html=True)
