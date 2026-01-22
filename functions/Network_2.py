import functions.functions_for_images as img_func
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

class Network2():

    def __init__(self, data):

        input_rows = 5
        input_cols = 5
        kernel_rows = 3
        kernel_cols = 3
        num_kernels = 16
        num_labels = 4
        batch_size = 1
        alpha = 0.001

        self.alpha = data['alpha']
        self.iterations = data['iterations']
        self.kernel_rows = data['kernel_rows']
        self.kernel_cols = data['kernel_cols']
        self.num_kernels = data['num_kernels']
        self.input_rows = data['input_rows']
        self.input_cols = data['input_cols']
        self.num_labels = data['num_labels']
        self.low_dir = data['low_dir']
        self.high_dir = data['high_dir']
        self.num_images = data['num_images']
        
        self.weights_0_1 = 0.2*np.random.random((input_rows * input_cols, 10)) - 0.1
        self.weights_1_2 = 0.2*np.random.random((10, num_labels)) - 0.1

    def relu(self, x):
        return np.maximum(0, x)

    def relu2deriv(self, x):
        return (x > 0).astype(np.float32)
    
    def training(self):
        num_images = 0

        N = self.input_rows
        B = N // 2

        for jj in os.listdir(self.low_dir):
            print(jj)

            if(num_images == self.num_images): break
            num_images += 1
            
            img1 = img_func.open_image(self.low_dir + "\\" + jj)
            img2 = img_func.open_image(self.high_dir + "\\" + jj)

            matrix1 = img_func.get_matrix_img(img1)
            matrix2 = img_func.get_matrix_img(img2)

            img3 = img_func.get_edges(matrix1, 2)

            for j in range(1):
                error = 0
                for i in range(B, img3.shape[0] - B):
                    for ii in range(B, img3.shape[1] - B):

                        x_matrix1 = i - B
                        y_matrix1 = ii - B

                        fragment1 = img3[x_matrix1:i + B + 1, y_matrix1:ii + B + 1]

                        x_matrix2 = x_matrix1 * 2
                        y_matrix2 = y_matrix1 * 2

                        fragment2 = matrix2[x_matrix2:x_matrix2 + 2, y_matrix2:y_matrix2 + 2]

                        for iii in range(3):
                            grey_fragment1 = img_func.get_grey_matri(fragment1, iii) / 255
                            grey_fragment2 = img_func.get_grey_matri(fragment2, iii) / 255

                            layer_0 = grey_fragment1[None, :, :]
                            labels = grey_fragment2[None, :, :]

                            layer_0 = layer_0.reshape(1, self.input_rows * self.input_cols)
                            labels = labels.reshape(1, self.num_labels)

                            layer_1 = self.relu(np.dot(layer_0, self.weights_0_1))
                            layer_2 = np.dot(layer_1, self.weights_1_2)

                            layer_2_delta = (labels - layer_2)
                            error += np.sum(layer_2_delta ** 2)
                            print(np.sum(layer_2_delta ** 2))

                            layer_1_delta = layer_2_delta.dot(self.weights_1_2.T) * self.relu2deriv(layer_1)

                            self.weights_1_2 += self.alpha * layer_1.T.dot(layer_2_delta)
                            self.weights_0_1 += self.alpha * layer_0.T.dot(layer_1_delta)

                print(j, error)

    def test(self, path):

        img = img_func.open_image(path)

        matrix = img_func.get_matrix_img(img)

        matrix3 = np.zeros((matrix.shape[0] * 2, matrix.shape[1] * 2, 3), dtype=np.uint8)

        img3 = img_func.get_edges(matrix, 2)

        N = self.input_rows
        B = N // 2

        for i in range(B, img3.shape[0] - B):
            for ii in range(B, img3.shape[1] - B):

                x_matrix1 = i - B
                y_matrix1 = ii - B

                x_matrix2 = x_matrix1 * 2
                y_matrix2 = y_matrix1 * 2

                fragment1 = img3[x_matrix1:i + B + 1, y_matrix1:ii + B + 1]

                for iii in range(3):
                    grey_fragment1 = img_func.get_grey_matri(fragment1, iii) / 255

                    layer_0 = grey_fragment1[None, :, :]

                    layer_0 = layer_0.reshape(1, self.input_rows * self.input_cols)

                    layer_1 = self.relu(np.dot(layer_0, self.weights_0_1))
                    layer_2 = np.dot(layer_1, self.weights_1_2)

                    layer_2 *= 255

                    img_fin = layer_2[0].reshape(2, 2).astype(np.uint8)

                    matrix3[x_matrix2][y_matrix2][iii] = img_fin[0][0]
                    matrix3[x_matrix2 + 1][y_matrix2][iii] = img_fin[1][0]
                    matrix3[x_matrix2][y_matrix2 + 1][iii] = img_fin[0][1]
                    matrix3[x_matrix2 + 1][y_matrix2 + 1][iii] = img_fin[1][1]

        img_func.show_imgs(matrix, matrix3)