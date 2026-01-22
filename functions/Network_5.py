import functions.functions_for_images as img_func
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

class Network5:
    def __init__(self, config):

        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.num_filters = config.get('num_filters', 32)
        self.kernel_size = config.get('kernel_size', 3)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.num_epochs = config.get('num_epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        
        self.low_dir = config['low_dir']
        self.high_dir = config['high_dir']

        self.weights_file = config['weights_file']
        
        self.conv1_weights = np.random.randn(self.kernel_size, self.kernel_size, 3, self.num_filters) * np.sqrt(2.0 / (self.kernel_size * self.kernel_size * 3))
        self.conv1_bias = np.zeros((1, 1, 1, self.num_filters))
        
        self.conv2_weights = np.random.randn(self.kernel_size, self.kernel_size, self.num_filters, self.num_filters) * np.sqrt(2.0 / (self.kernel_size * self.kernel_size * self.num_filters))
        self.conv2_bias = np.zeros((1, 1, 1, self.num_filters))
        
        conv_output_size = self.input_size
        flattened_size = conv_output_size * conv_output_size * self.num_filters
        
        self.fc_weights = np.random.randn(flattened_size, self.output_size * self.output_size * 3) * np.sqrt(2.0 / flattened_size)
        self.fc_bias = np.zeros((1, self.output_size * self.output_size * 3))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def conv2d(self, input_data, weights, bias, padding='same'):

        batch_size, in_height, in_width, in_channels = input_data.shape
        kernel_h, kernel_w, _, out_channels = weights.shape
        
        if padding == 'same':
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            padded_input = np.pad(input_data, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            padded_input = input_data
            pad_h = pad_w = 0
        
        out_height = in_height
        out_width = in_width
        
        output = np.zeros((batch_size, out_height, out_width, out_channels))
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i
                w_start = j
                h_end = h_start + kernel_h
                w_end = w_start + kernel_w
                
                input_slice = padded_input[:, h_start:h_end, w_start:w_end, :]
                
                for k in range(out_channels):
                    output[:, i, j, k] = np.sum(
                        input_slice * weights[:, :, :, k],
                        axis=(1, 2, 3)
                    ) + bias[0, 0, 0, k]
        
        return output
    
    def conv2d_backward(self, input_data, weights, d_output, padding='same'):

        batch_size, in_height, in_width, in_channels = input_data.shape
        kernel_h, kernel_w, _, out_channels = weights.shape
        
        if padding == 'same':
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            padded_input = np.pad(input_data, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            padded_input = input_data
            pad_h = pad_w = 0
        
        d_weights = np.zeros_like(weights)
        d_bias = np.zeros((1, 1, 1, out_channels))
        d_input = np.zeros_like(padded_input)
        
        out_height = in_height
        out_width = in_width
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i
                w_start = j
                h_end = h_start + kernel_h
                w_end = w_start + kernel_w
                
                input_slice = padded_input[:, h_start:h_end, w_start:w_end, :]
                
                for k in range(out_channels):
                    d_weights[:, :, :, k] += np.sum(
                        input_slice * d_output[:, i:i+1, j:j+1, k:k+1],
                        axis=0
                    )
                    
                    d_input[:, h_start:h_end, w_start:w_end, :] += (
                        weights[:, :, :, k] * d_output[:, i:i+1, j:j+1, k:k+1]
                    )
                
                d_bias[0, 0, 0, :] += np.sum(d_output[:, i, j, :], axis=0)
        
        if padding == 'same':
            d_input = d_input[:, pad_h:-pad_h or None, pad_w:-pad_w or None, :]
        
        return d_input, d_weights, d_bias
    
    def forward(self, x):

        self.conv1_out = self.conv2d(x, self.conv1_weights, self.conv1_bias, padding='same')
        self.conv1_activated = self.relu(self.conv1_out)
        

        self.conv2_out = self.conv2d(self.conv1_activated, self.conv2_weights, self.conv2_bias, padding='same')
        self.conv2_activated = self.relu(self.conv2_out)
        

        self.flattened = self.conv2_activated.reshape(self.conv2_activated.shape[0], -1)
        

        self.fc_out = np.dot(self.flattened, self.fc_weights) + self.fc_bias
        

        output_shape = (x.shape[0], self.output_size, self.output_size, 3)
        output = self.fc_out.reshape(output_shape)
        

        return np.clip(output, 0, 1)
    
    def backward(self, x, y_true, y_pred):

        batch_size = x.shape[0]
        
        d_output = 2 * (y_pred - y_true) / batch_size
        
        d_fc = d_output.reshape(batch_size, -1)
        
        d_fc_weights = np.dot(self.flattened.T, d_fc)
        d_fc_bias = np.sum(d_fc, axis=0, keepdims=True)
        d_flattened = np.dot(d_fc, self.fc_weights.T)
        
        d_conv2_activated = d_flattened.reshape(self.conv2_activated.shape)
        
        d_conv2_out = d_conv2_activated * self.relu_derivative(self.conv2_out)
        
        d_conv1_activated, d_conv2_weights, d_conv2_bias = self.conv2d_backward(
            self.conv1_activated, self.conv2_weights, d_conv2_out, padding='same'
        )
        
        d_conv1_out = d_conv1_activated * self.relu_derivative(self.conv1_out)
        
        _, d_conv1_weights, d_conv1_bias = self.conv2d_backward(
            x, self.conv1_weights, d_conv1_out, padding='same'
        )
        
        self.conv1_weights -= self.learning_rate * d_conv1_weights
        self.conv1_bias -= self.learning_rate * d_conv1_bias
        self.conv2_weights -= self.learning_rate * d_conv2_weights
        self.conv2_bias -= self.learning_rate * d_conv2_bias
        self.fc_weights -= self.learning_rate * d_fc_weights
        self.fc_bias -= self.learning_rate * d_fc_bias
        
        return np.mean(np.square(y_pred - y_true))
    
    def wait_for_stop(self, timeout=10):

        import threading
        import sys
        
        stop_command = False
        
        def listen_for_stop():

            nonlocal stop_command
            print(f"\nОжидание команды 'stop' в течение {timeout} секунд...")
            print("Введите 'stop' и нажмите Enter для остановки обучения")
            
            def input_thread():
                nonlocal stop_command
                try:
                    user_input = input().strip().lower()
                    if user_input == 'stop':
                        stop_command = True
                except:
                    pass
            
            input_thread_obj = threading.Thread(target=input_thread, daemon=True)
            input_thread_obj.start()
            
            input_thread_obj.join(timeout=timeout)
            
            if input_thread_obj.is_alive():

                try:
                    import select
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.readline()
                except:
                    pass
                print(f"Таймаут {timeout} секунд истек. Продолжаем обучение...")
        
        listen_for_stop()
        
        return stop_command

    def train(self, num_images=None, avto_save=True):

        low_images = sorted([f for f in os.listdir(self.low_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        high_images = sorted([f for f in os.listdir(self.high_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        if num_images:
            low_images = low_images[:num_images]
            high_images = high_images[:num_images]
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            processed_images = 0
            
            for low_img_name, high_img_name in zip(low_images, high_images):
                img_loss = 0
                processed_image = 0

                img1 = Image.open(os.path.join(self.low_dir, low_img_name)).convert('RGB')
                img2 = Image.open(os.path.join(self.high_dir, high_img_name)).convert('RGB')
                
                matrix1 = img_func.get_matrix_img(img1)
                matrix2 = img_func.get_matrix_img(img2)
                
                expanded1 = img_func.expand_matrix(matrix1, self.input_size, self.input_size)
                expanded2 = img_func.expand_matrix(matrix2, self.output_size, self.output_size)
                
                fragments1 = img_func.cut_matrix(expanded1, self.input_size, self.input_size)
                fragments2 = img_func.cut_matrix(expanded2, self.output_size, self.output_size)
                
                for i in range(0, len(fragments1), self.batch_size):
                    batch_input = fragments1[i:i+self.batch_size] / 255.0
                    batch_target = fragments2[i:i+self.batch_size] / 255.0
                    
                    predictions = self.forward(batch_input)
                    
                    loss = self.backward(batch_input, batch_target, predictions)
                    
                    img_loss += loss
                    processed_image += len(batch_input)

                print(f"Изображение {low_img_name}, Loss: {img_loss / (processed_image / self.batch_size) if processed_image > 0 else 0}")

                if avto_save:
                    self.save_weights_npz(self.weights_file)

                    if self.wait_for_stop(10):
                        print("Получена команда остановки. Завершаем обучение...")
                        return  


                total_loss += img_loss
                processed_images += processed_image
            
            avg_loss = total_loss / (processed_images / self.batch_size) if processed_images > 0 else 0
            print(f"Эпоха {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
    
    def enhance_image(self, image_path, save_path=None):

        img = Image.open(image_path).convert('RGB')
        original_matrix = img_func.get_matrix_img(img)
        
        original_height, original_width, _ = original_matrix.shape
        
        expanded = img_func.expand_matrix(original_matrix, self.input_size, self.input_size)
        fragments = img_func.cut_matrix(expanded, self.input_size, self.input_size)
        
        enhanced_fragments = []
        
        for i in range(0, len(fragments), self.batch_size):
            batch = fragments[i:i+self.batch_size] / 255.0
            enhanced_batch = self.forward(batch)
            enhanced_fragments.append(enhanced_batch * 255)
        
        enhanced_fragments = np.vstack(enhanced_fragments)
        enhanced_expanded = img_func.collect_img(
            enhanced_fragments, 
            expanded.shape[0] // self.input_size,
            expanded.shape[1] // self.input_size
        )
        
        enhanced_matrix = img_func.reduction_matrix(
            enhanced_expanded, 
            original_height * (self.output_size // self.input_size),
            original_width * (self.output_size // self.input_size)
        )
        
        enhanced_matrix = np.clip(enhanced_matrix, 0, 255).astype(np.uint8)
        
        enhanced_img = Image.fromarray(enhanced_matrix)
        
        if save_path:
            enhanced_img.save(save_path)
        
        return enhanced_matrix
    
    def save_weights_npz(self, filepath):

        np.savez(
            filepath,
            conv1_weights=self.conv1_weights,
            conv1_bias=self.conv1_bias,
            conv2_weights=self.conv2_weights,
            conv2_bias=self.conv2_bias,
            fc_weights=self.fc_weights,
            fc_bias=self.fc_bias,
            config=np.array([self.input_size, self.output_size, self.num_filters, 
                           self.kernel_size, self.learning_rate, self.num_epochs, 
                           self.batch_size])
        )
        
        print(f"Веса успешно сохранены в {filepath}")
    
    def load_weights_npz(self, filepath):

        data = np.load(filepath, allow_pickle=True)
        
        self.conv1_weights = data['conv1_weights']
        self.conv1_bias = data['conv1_bias']
        self.conv2_weights = data['conv2_weights']
        self.conv2_bias = data['conv2_bias']
        self.fc_weights = data['fc_weights']
        self.fc_bias = data['fc_bias']
        
        if 'config' in data:
            config = data['config']
            if len(config) >= 7:
                print(f"Конфигурация сохраненной модели: input_size={config[0]}, "
                      f"output_size={config[1]}, num_filters={config[2]}, "
                      f"kernel_size={config[3]}")
        
        print(f"Веса успешно загружены из {filepath}")

