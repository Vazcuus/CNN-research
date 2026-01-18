import functions.Network_1 as net_1
import functions.Network_2 as net_2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

choise = int(input("Выберите версию сети:"))

if choise == 1:
  x = 5
  y = 5

  data = {
    'alpha' : 0.001,
    'iterations' : 1,
    'pixels_per_image' : x * y,
    'kernel_rows' : 3,
    'kernel_cols' : 3,
    'num_kernels' : 9,
    'input_rows' : x,
    'input_cols' : y,
    'output_rows' : x * 2,
    'output_cols' : y * 2,
    'num_labels' : ((x * 2) ** 2),
    'low_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\Low_Resolution",
    'high_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\High_Resolution",
    #'low_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\NL",
    #'high_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\NH",
    'num_images' : 1
  }

  N = net_1.Network1(data)
  print(N.kernels)
  print("___________________________________________________")
  N.training()
  print("___________________________________________________")
  N.test("D:\\projects\\study\\5th_semester\\UIRS\\data\\Low_Resolution\\image_000001.jpg")
  N.test("D:\\projects\\study\\5th_semester\\UIRS\\data\\NL\\img2.png")
  print(N.weights_l_2)
  print('end')

if choise == 2:
  data = {
    'alpha' : 0.001,
    'iterations' : 1,
    'kernel_rows' : 3,
    'kernel_cols' : 3,
    'num_kernels' : 9,
    'input_rows' : 5,
    'input_cols' : 5,
    'num_labels' : 4,
    'low_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\Low_Resolution",
    'high_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\High_Resolution",
    #'low_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\NL",
    #'high_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\NH",
    'num_images' : 1
  }
  N = net_2.Network2(data)
  print("___________________________________________________")
  N.training()
  print("___________________________________________________")
  N.test("D:\\projects\\study\\5th_semester\\UIRS\\data\\Low_Resolution\\image_000001.jpg")
  N.test("D:\\projects\\study\\5th_semester\\UIRS\\data\\NL\\img2.png")
  print('end')

