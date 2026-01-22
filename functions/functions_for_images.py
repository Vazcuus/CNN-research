from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def open_image(file_path=None):
    if file_path is None:
        file_path = input("Введите путь к изображению: ")
        
    img = Image.open(file_path).convert('RGB')
    return img

def get_matrix_img(img):
  img_matrix = np.array(img)
  return img_matrix

def expand_matrix(matrix, x, y):
  if(matrix.shape[0] % x != 0 or matrix.shape[1] % y != 0):
    pixel = np.array([0,  0,  0], dtype=np.uint8)
    column = np.full((matrix.shape[0], 1, 3), pixel)
    for i in range(y - matrix.shape[1] % y):
      matrix = np.append(matrix, column, axis = 1)
    row = np.full((1, matrix.shape[1], 3), pixel)
    for i in range(x - matrix.shape[0] % x):
      matrix = np.append(matrix, row, axis = 0)
  return matrix

def reduction_matrix(matrix, x, y):
  for i in range(matrix.shape[0] - 1, x - 1, -1):
    matrix = np.delete(matrix, i, axis=0)
  for i in range(matrix.shape[1] - 1, y - 1, -1):
    matrix = np.delete(matrix, i, axis=1)
  return matrix

def cut_matrix(matrix, x, y):
  fragments = np.zeros((int(matrix.shape[0] / x) * int(matrix.shape[1] / y), x, y, matrix.shape[2]), dtype=np.uint8)
  iii = 0
  for i in range(0, matrix.shape[0], x):
    for ii in range(0, matrix.shape[1], y):
      fragments[iii] = matrix[i:i+x, ii:ii+y]
      iii += 1
  return fragments

def filter_matrix(matrix, chanal):
  filtered = np.zeros(matrix.shape, dtype=np.uint8)
  for i in range(filtered.shape[0]):
    for ii in range(filtered.shape[1]):
      filtered[i][ii][chanal] = matrix[i][ii][chanal]
  return filtered

def get_vector_img(img, chanal):
  vector = np.zeros(img.shape[0] * img.shape[1], dtype=np.uint8)
  iii = 0
  for i in range(img.shape[0]):
    for ii in range(img.shape[1]):
      vector[iii] = img[i][ii][chanal]
      iii += 1
  return vector

def get_img_vector(vector, x, y, chanal):
  img = np.zeros((x, y, 3), dtype=np.uint8)
  iii = 0
  for i in range(x):
    for ii in range(y):
      img[i][ii][chanal] = vector[iii]
      iii += 1
  return img

def merge_filtered_img(filtereds_img):
  img = np.zeros(filtereds_img[0].shape, dtype=np.uint8)
  for i in range(filtereds_img[0].shape[0]):
    for ii in range(filtereds_img[0].shape[1]):
      for iii in range(3):
        img[i][ii][iii] = filtereds_img[iii][i][ii][iii]
  return img

def collect_img(fragments, x, y):
  img = np.zeros((int(fragments[0].shape[0] * x), int(fragments[0].shape[1] * y), 3), dtype=np.uint8)
  j = 0
  jj = 0
  for i in range(fragments.shape[0]):
    for ii in range(fragments[0].shape[0]):
      for iii in range(fragments[0].shape[1]):
        for iiii in range(3):
          img[j + ii][jj + iii][iiii] = fragments[i][ii][iii][iiii]
    jj += fragments[0].shape[1]
    if(jj == y * fragments[0].shape[1]):
      j += fragments[0].shape[0]
      jj = 0
  return img

def show_img(img):
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')  
    plt.show()

def show_imgs(img1, img2):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img1)
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def get_grey_matri(img, chanal):
  grey_matrix = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
  for i in range(img.shape[0]):
    for ii in range(img.shape[1]):
      grey_matrix[i][ii] = img[i][ii][chanal]
  return grey_matrix

def get_edges(img, num):
  new_img = np.empty((img.shape[0] + num * 2, img.shape[1] + num * 2, img.shape[2]), dtype=np.uint8)
  for i in range(0, img.shape[0]):
    for ii in range(0, img.shape[1]):
      new_img[i + num][ii + num] = img[i][ii]

  for i in range(num):
    for ii in range(img.shape[1]):
      new_img[i][ii + num] = img[0][ii]

  for i in range(num):
    for ii in range(img.shape[1]):
      new_img[i][ii + num] = img[0][ii]
      new_img[new_img.shape[0] - i - 1][ii + num] = img[-1][ii]

  for i in range(num):
    for ii in range(img.shape[0]):
      new_img[ii + num][i] = img[ii][0]
      new_img[ii + num][new_img.shape[1] - i - 1] = img[ii][-1]

  for i in range(num):
    for ii in range(num):
      new_img[i][ii] = img[0][0]
      new_img[i][new_img.shape[1] - ii - 1] = img[0][-1]
      new_img[new_img.shape[0] - i - 1][ii] = img[-1][0]
      new_img[new_img.shape[0] - i - 1][new_img.shape[1] - ii - 1] = img[-1][-1]

  return new_img

