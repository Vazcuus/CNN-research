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
    self.kernels = np.random.randn(self.kernel_rows*self.kernel_cols, self.num_kernels)* np.sqrt(2.0 / self.kernel_rows*self.kernel_cols)
    self.weights_l_2 = np.random.randn(self.hidden_size, self.num_labels) * np.sqrt(2.0 / self.hidden_size)

  def get_image_section_single(self, layer, row_from, row_to, col_from, col_to):
    section = layer[row_from:row_to, col_from:col_to]  
    return section.reshape(1, row_to - row_from, col_to - col_from)
  
  def relu(self, x):
    return np.maximum(0, x)
  
  def relu_deriv(self, x):
    return (x >= 0).astype(float)
  
  def sigmoid(self, x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))
  
  def sigmoid_derivative(self, x):
    s = self.sigmoid(x)
    return s * (1 - s)
  
  def training(self):
    num_images = 0

    for i in os.listdir(self.low_dir):
      print(i)

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
          list1[ii][iii] = img_func.get_vector_img(fragments1[ii], iii) / 255.0
          list2[ii][iii] = img_func.get_vector_img(fragments2[ii], iii) / 255.0

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

            layer_l = self.sigmoid(kernel_output.reshape(-1))  

            layer_2 = np.dot(layer_l, self.weights_l_2)

            layer_2_delta = (list2[iii][iiii] - layer_2)
            #layer_2_delta = np.clip(layer_2_delta, -1, 1)
            layer_l_delta = layer_2_delta.dot(self.weights_l_2.T) * self.sigmoid_derivative(layer_l)

            kernel_delta = layer_l_delta.reshape(kernel_output.shape)
            kernels_gradient = flattened_input.T.dot(kernel_delta)
            #kernels_gradient = np.clip(kernels_gradient, -1, 1)

            self.kernels -= self.alpha * kernels_gradient

            self.weights_l_2 += self.alpha * np.outer(layer_l, layer_2_delta)

  def test(self, path):
    img = img_func.open_image(path)

    matrix = img_func.get_matrix_img(img)

    expanded_matrix = img_func.expand_matrix(matrix, self.input_rows, self.input_cols)

    fragments = img_func.cut_matrix(expanded_matrix, self.input_rows, self.input_cols)

    list1 = np.empty((fragments.shape[0], 3), dtype=object)
    list2 = np.empty((fragments.shape[0], 3), dtype=object)

    for i in range(fragments.shape[0]):
      for ii in range(3):
        list1[i][ii] = img_func.get_vector_img(fragments[i], ii) / 255.0

    for i in range(list1.shape[0]):
      for ii in range(3):
        layer_0 = list1[i][ii].reshape((self.input_rows, self.input_cols)) 
        
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

        layer_l = self.sigmoid(kernel_output.reshape(-1))  

        layer_2 = np.dot(layer_l, self.weights_l_2)

        layer_2 *= 255 

        list2[i][ii] = layer_2
    
    filtered_fragments_list = np.empty((list2.shape[0], 3), dtype=object)
    for i in range(fragments.shape[0]):
      for ii in range(3):
        filtered_fragments_list[i][ii] = img_func.get_img_vector(list2[i][ii], self.output_rows, self.output_cols, ii)
    

    fragments_list = np.empty((filtered_fragments_list.shape[0]), dtype=object)
    for i in range(filtered_fragments_list.shape[0]):
      fragments_list[i] = img_func.merge_filtered_img(filtered_fragments_list[i])

    expanded_matrix_list = img_func.collect_img(fragments_list, expanded_matrix.shape[0] / self.input_rows, expanded_matrix.shape[1] / self.input_cols)

    matrix_list = img_func.reduction_matrix(expanded_matrix_list, matrix.shape[0] * 2, matrix.shape[1] * 2)

    img_func.show_img(matrix_list)