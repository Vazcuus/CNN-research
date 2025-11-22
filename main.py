import functions.functions_for_images as img_func
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

x = int(input("Введите ширену фрагмента: "))
y = int(input("Введите высоту фрагмента: "))

img1 = img_func.open_image()

matrix = img_func.get_matrix_img(img1)
expanded_matrix = img_func.expand_matrix(matrix, x, y)
fragments = img_func.cut_matrix(expanded_matrix, x, y)

filtered_fragments = np.empty((fragments.shape[0], 3), dtype=object)
for i in range(fragments.shape[0]):
  for ii in range(3):
    filtered_fragments[i][ii] = img_func.filter_matrix(fragments[i], ii)

list = np.empty((filtered_fragments.shape[0], filtered_fragments.shape[1]), dtype=object)
for i in range(fragments.shape[0]):
  for ii in range(3):
    list[i][ii] = img_func.get_vector_img(fragments[i], ii)

filtered_fragments_list = np.empty((filtered_fragments.shape[0], filtered_fragments.shape[1]), dtype=object)
for i in range(fragments.shape[0]):
  for ii in range(3):
    filtered_fragments_list[i][ii] = img_func.get_img_vector(list[i][ii], filtered_fragments[0][0].shape[0], filtered_fragments[0][0].shape[1], ii)

fragments_list = np.empty((filtered_fragments_list.shape[0]), dtype=object)
for i in range(filtered_fragments_list.shape[0]):
  fragments_list[i] = img_func.merge_filtered_img(filtered_fragments_list[i])

expanded_matrix_list = img_func.collect_img(fragments_list, expanded_matrix.shape[0] / x, expanded_matrix.shape[1] / y)

matrix_list = img_func.reduction_matrix(expanded_matrix_list, matrix.shape[0], matrix.shape[1])


img = Image.fromarray(matrix, mode='RGB')
img_func.show_img(img)

img = Image.fromarray(expanded_matrix, mode='RGB')
img_func.show_img(img)

img = Image.fromarray(fragments[0], mode='RGB')
img_func.show_img(img)

img = Image.fromarray(filtered_fragments[0][0], mode='RGB')
img_func.show_img(img)

print(list[0][0])

img = Image.fromarray(filtered_fragments_list[0][0], mode='RGB')
img_func.show_img(img)

img = Image.fromarray(fragments_list[0], mode='RGB')
img_func.show_img(img)

img = Image.fromarray(expanded_matrix_list, mode='RGB')
img_func.show_img(img)

img = Image.fromarray(matrix_list, mode='RGB')
img_func.show_img(img)