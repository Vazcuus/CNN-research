import functions.functions_for_images as img_func
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

class Network1():

  def __init__(self, data):
    self.alpha = data['alpha']
    self.iterations = data['iterations']
    self.pixels_per_image = data['pixels_per_image']
    self.kernel_rows = data['kernel_rows']
    self.kernel_cols = data['kernel_cols']
    self.num_kernels = data['num_kernels']
    self.input_rows = data['input_rows']
    self.input_cols = data['input_cols']
    self.output_rows = data['output_rows']
    self.output_cols = data['output_cols']
    self.num_labels = data['num_labels']
    self.low_dir = data['low_dir']
    self.high_dir = data['high_dir']
    self.num_images = data['num_images']
    
    self.hidden_size = ((self.input_rows - self.kernel_rows) * (self.input_cols - self.kernel_cols)) * self.num_kernels
    self.kernels = 0.02*np.random.random((self.kernel_rows*self.kernel_cols, self.num_kernels))-0.01
    self.weights_l_2 = 0.2*np.random.random((self.hidden_size, self.num_labels)) - 0.1

  def get_image_section_single(self, layer, row_from, row_to, col_from, col_to):
    section = layer[row_from:row_to, col_from:col_to]  
    return section.reshape(1, row_to - row_from, col_to - col_from)
  
  def relu(self, x):
    return np.maximum(0, x)
  
  def relu_deriv(self, x):
    return (x >= 0).astype(float)
  
  def training(self):
    num_images = 0

    for i in os.listdir(self.low_dir):
      if(num_images == self.num_images): break
      num_images += 1
      
      img1 = img_func.open_image(self.low_dir + "\\" + i)
      img2 = img_func.open_image(self.high_dir + "\\" + i)

      matrix1 = img_func.get_matrix_img(img1)
      matrix2 = img_func.get_matrix_img(img2)

      expanded_matrix1 = img_func.expand_matrix(matrix1, self.input_rows, self.input_cols)
      expanded_matrix2 = img_func.expand_matrix(matrix2, self.output_rows, self.output_cols)

      fragments1 = img_func.cut_matrix(expanded_matrix1, self.input_rows, self.input_cols)
      fragments2 = img_func.cut_matrix(expanded_matrix2, self.output_rows, self.output_rows)

      list1 = np.empty((fragments1.shape[0], fragments1.shape[1]), dtype=object)
      list2 = np.empty((fragments2.shape[0], fragments2.shape[1]), dtype=object)
      for ii in range(fragments1.shape[0]):
        for iii in range(3):
          list1[ii][iii] = img_func.get_vector_img(fragments1[ii], iii)
          list2[ii][iii] = img_func.get_vector_img(fragments2[ii], iii)

      for ii in range(self.iterations):
        for iii in range(list1.shape[0]):
          for iiii in range(3):
            #print("image:", i, "  iteration:", ii, "  fragment:", iii * 3 + iiii)
            layer_0 = list1[iii][iiii].reshape((self.input_rows, self.input_cols)) 
            
            sects = list()
            for row_start in range(layer_0.shape[0] - self.kernel_rows):  
                for col_start in range(layer_0.shape[1] - self.kernel_cols):  
                    sect = self.get_image_section_single(layer_0,  
                                                         row_start, 
                                                         row_start + self.kernel_rows, 
                                                         col_start, 
                                                         col_start + self.kernel_cols)
                    sects.append(sect)

            expanded_input = np.concatenate(sects, axis=0)  
            es = expanded_input.shape
            flattened_input = expanded_input.reshape(es[0], -1)

            kernel_output = flattened_input.dot(self.kernels) 

            layer_l = self.relu(kernel_output.reshape(-1))  

            layer_2 = np.dot(layer_l, self.weights_l_2)

            layer_2_delta = (list2[iii][iiii] - layer_2)
            layer_l_delta = layer_2_delta.dot(self.weights_l_2.T) * self.relu_deriv(layer_l)

            kernel_delta = layer_l_delta.reshape(kernel_output.shape)
            kernels_gradient = flattened_input.T.dot(kernel_delta)
            self.kernels -= self.alpha * kernels_gradient

            self.weights_l_2 += self.alpha * np.outer(layer_l, layer_2_delta)


x = 25
y = 25

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
  'num_images' : 1
}

N = Network1(data)
print(N.kernels)
print("___________________________________________________")
N.training()
print('end')

