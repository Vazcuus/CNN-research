import functions.Network_1 as net_1
import functions.Network_2 as net_2
import functions.Network_3 as net_3
import functions.Network_4 as net_4
import functions.Network_5 as net_5
import functions.functions_for_images as img_func
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


if choise == 3:
  config = {
    'input_size': 8,
    'output_size': 16,
    'num_filters': 16,
    'kernel_size': 3,
    'learning_rate': 0.0005,
    'num_epochs': 10,
    'batch_size': 1,
    'low_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\Low_Resolution",
    'high_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\High_Resolution"
  }

  model = net_3.EnhancedCNN(config)
  model.train(num_images=1)


  enhanced = model.enhance_image('D:\\projects\\study\\5th_semester\\UIRS\\data\\Low_Resolution\\image_000001.jpg',
                                  'D:\\projects\\study\\5th_semester\\UIRS\\data\\enhanced_image.jpg')
  

if choise == 4:
  config = {
      'input_size': 8,
      'output_size': 16,
      'num_filters': 32,
      'kernel_size': 3,
      'learning_rate': 0.001,
      'num_epochs': 10,
      'batch_size': 16,
      'low_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\Low_Resolution",
      'high_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\High_Resolution",
      'weights_file': 'my_cnn_weights.npz',  
      'wait_time': 10 
  }

  cnn = net_4.EnhancedCNN(config)

  #cnn.train(num_images=1, auto_save=True)
  

  if not cnn.stop_training:
      enhanced = cnn.enhance_image('D:\\projects\\study\\5th_semester\\UIRS\\data\\Low_Resolution\\image_000001.jpg', 
                                   'D:\\projects\\study\\5th_semester\\UIRS\\data\\enhanced_image_net4.jpg')
    
      img_func.show_imgs(
          np.array(Image.open('D:\\projects\\study\\5th_semester\\UIRS\\data\\Low_Resolution\\image_000001.jpg').convert('RGB')),
          enhanced
      )

if choise == 5:
  config = {
    'input_size': 8,
    'output_size': 16,
    'num_filters': 16,
    'kernel_size': 3,
    'learning_rate': 0.0005,
    'num_epochs': 20,
    'batch_size': 1,
    'low_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\Low_Resolution",
    'high_dir' : "D:\\projects\\study\\5th_semester\\UIRS\\data\\High_Resolution",
    'weights_file' : "model_weights"
  }

  model = net_5.EnhancedCNN(config)
  #model.load_weights_npz('model_weights.npz')

  model.train(num_images=1)

  model.save_weights_npz('model_weights_for_1')
  #model.load_weights_npz('model_weights.npz')

  enhanced = model.enhance_image('D:\\projects\\study\\5th_semester\\UIRS\\data\\High_Resolution\\image_000001.jpg',
                                  #'D:\\projects\\study\\5th_semester\\UIRS\\data\\enhanced_image3_net5.jpg',
                                  'D:\\projects\\study\\5th_semester\\UIRS\\data\\enhanced_image3_net5.jpg')