import functions.functions_for_images as img_func
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import time
import threading

class EnhancedCNN:
    def __init__(self, config):

        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.num_filters = config.get('num_filters', 16) 
        self.kernel_size = config.get('kernel_size', 3)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.num_epochs = config.get('num_epochs', 10)  
        self.batch_size = config.get('batch_size', 8)     
        
        self.low_dir = config['low_dir']
        self.high_dir = config['high_dir']
        self.weights_file = config.get('weights_file', 'cnn_weights.npz')
        self.wait_time = config.get('wait_time', 10)
        
        self.stop_training = False
        
        if os.path.exists(self.weights_file):
            print(f"Загружаем веса из файла: {self.weights_file}")
            self.load_weights(self.weights_file)
        else:
            print(f"Файл весов не найден. Инициализируем новые веса.")
            self.initialize_weights()
            self.save_weights(self.weights_file)
            
        print(f"Архитектура сети:")
        print(f"  Вход: {self.input_size}x{self.input_size}x3")
        print(f"  Conv1: {self.kernel_size}x{self.kernel_size}x3x{self.num_filters}")
        print(f"  Conv2: {self.kernel_size}x{self.kernel_size}x{self.num_filters}x{self.num_filters}")
        print(f"  Выход: {self.output_size}x{self.output_size}x3")
    
    def initialize_weights(self):

        self.conv_weights = np.random.randn(self.kernel_size, self.kernel_size, 3, self.num_filters) * np.sqrt(2.0 / (self.kernel_size * self.kernel_size * 3))
        self.conv_bias = np.zeros((1, 1, 1, self.num_filters))
        
        conv_output_size = self.input_size - self.kernel_size + 1
        flattened_size = conv_output_size * conv_output_size * self.num_filters
        
        self.fc_weights = np.random.randn(flattened_size, self.output_size * self.output_size * 3) * np.sqrt(2.0 / flattened_size)
        self.fc_bias = np.zeros((1, self.output_size * self.output_size * 3))
        
        print(f"  Conv выход: {conv_output_size}x{conv_output_size}x{self.num_filters}")
        print(f"  Flattened: {flattened_size}")
    
    def save_weights(self, filepath=None):
        if filepath is None:
            filepath = self.weights_file
        
        np.savez(
            filepath,
            conv_weights=self.conv_weights,
            conv_bias=self.conv_bias,
            fc_weights=self.fc_weights,
            fc_bias=self.fc_bias,
            config={
                'input_size': self.input_size,
                'output_size': self.output_size,
                'num_filters': self.num_filters,
                'kernel_size': self.kernel_size,
                'learning_rate': self.learning_rate
            }
        )
        print(f"Веса сохранены в: {filepath}")
    
    def load_weights(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        
        self.conv_weights = data['conv_weights']
        self.conv_bias = data['conv_bias']
        self.fc_weights = data['fc_weights']
        self.fc_bias = data['fc_bias']
        
        if 'config' in data:
            config = data['config'].item()
            print(f"Загружена конфигурация: вход {config['input_size']}x{config['input_size']}, выход {config['output_size']}x{config['output_size']}")
    
    def relu(self, x):

        return np.maximum(0, x)
    
    def relu_derivative(self, x):

        return (x > 0).astype(float)
    
    def conv2d_forward(self, input_data):

        batch_size, in_height, in_width, in_channels = input_data.shape
        kernel_h, kernel_w, in_channels, out_channels = self.conv_weights.shape
        
        out_height = in_height - kernel_h + 1
        out_width = in_width - kernel_w + 1
        
        output = np.zeros((batch_size, out_height, out_width, out_channels))
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_height):
                    for j in range(out_width):

                        window = input_data[b, i:i+kernel_h, j:j+kernel_w, :]

                        output[b, i, j, oc] = np.sum(window * self.conv_weights[:, :, :, oc]) + self.conv_bias[0, 0, 0, oc]
        
        return output
    
    def conv2d_backward(self, input_data, d_output):

        batch_size, in_height, in_width, in_channels = input_data.shape
        kernel_h, kernel_w, in_channels, out_channels = self.conv_weights.shape
        _, out_height, out_width, _ = d_output.shape
        
        d_conv_weights = np.zeros_like(self.conv_weights)
        d_conv_bias = np.zeros_like(self.conv_bias)
        d_input = np.zeros_like(input_data)
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_height):
                    for j in range(out_width):

                        window = input_data[b, i:i+kernel_h, j:j+kernel_w, :]
                        grad = d_output[b, i, j, oc]
                        
                        d_conv_weights[:, :, :, oc] += window * grad
                        
                        d_conv_bias[0, 0, 0, oc] += grad
                        
                        d_input[b, i:i+kernel_h, j:j+kernel_w, :] += self.conv_weights[:, :, :, oc] * grad
        
        return d_input, d_conv_weights, d_conv_bias
    
    def forward(self, x):

        self.conv_out = self.conv2d_forward(x)
        self.conv_activated = self.relu(self.conv_out)
        
        self.flattened = self.conv_activated.reshape(self.conv_activated.shape[0], -1)
        
        self.fc_out = np.dot(self.flattened, self.fc_weights) + self.fc_bias
        
        output_shape = (x.shape[0], self.output_size, self.output_size, 3)
        output = self.fc_out.reshape(output_shape)
        
        output = 1 / (1 + np.exp(-np.clip(output, -10, 10)))
        
        return output
    
    def backward(self, x, y_true, y_pred):

        batch_size = x.shape[0]
        
        d_output = 2 * (y_pred - y_true) / batch_size
        
        d_output = d_output * y_pred * (1 - y_pred)
        
        d_fc = d_output.reshape(batch_size, -1)
        
        d_fc_weights = np.dot(self.flattened.T, d_fc)
        d_fc_bias = np.sum(d_fc, axis=0, keepdims=True)
        d_flattened = np.dot(d_fc, self.fc_weights.T)
        
        d_conv_activated = d_flattened.reshape(self.conv_activated.shape)
        
        d_conv_out = d_conv_activated * self.relu_derivative(self.conv_out)
        
        d_input, d_conv_weights, d_conv_bias = self.conv2d_backward(x, d_conv_out)
        
        self.conv_weights -= self.learning_rate * d_conv_weights
        self.conv_bias -= self.learning_rate * d_conv_bias
        self.fc_weights -= self.learning_rate * d_fc_weights
        self.fc_bias -= self.learning_rate * d_fc_bias
        
        return np.mean(np.square(y_pred - y_true))
    
    def wait_for_stop_command(self, seconds=10):

        print(f"\nОжидание {seconds} секунд для команды остановки...")
        print("Нажмите Enter для продолжения или введите 'stop' для остановки обучения")
        
        stop_event = threading.Event()
        
        def input_thread():
            try:
                import sys
                if sys.version_info[0] < 3:
                    user_input = raw_input()
                else:
                    user_input = input()
                    
                if user_input.lower().strip() == 'stop':
                    self.stop_training = True
                    print("Получена команда остановки!")
                stop_event.set()
            except EOFError:
                stop_event.set()
        
        input_thread_obj = threading.Thread(target=input_thread)
        input_thread_obj.daemon = True
        input_thread_obj.start()
        
        for i in range(seconds):
            if self.stop_training:
                break
            if stop_event.is_set():
                break
            print(f"Осталось {seconds - i} секунд...", end='\r')
            time.sleep(1)
        
        print()  
        return self.stop_training
    
    def process_single_image(self, low_img_name, high_img_name):

        try:
            print(f"  Загрузка изображений: {low_img_name}")
            
            img1 = Image.open(os.path.join(self.low_dir, low_img_name)).convert('RGB')
            img2 = Image.open(os.path.join(self.high_dir, high_img_name)).convert('RGB')
            
            matrix1 = img_func.get_matrix_img(img1)
            matrix2 = img_func.get_matrix_img(img2)
            
            expanded1 = img_func.expand_matrix(matrix1, self.input_size, self.input_size)
            expanded2 = img_func.expand_matrix(matrix2, self.output_size, self.output_size)
            
            fragments1 = img_func.cut_matrix(expanded1, self.input_size, self.input_size)
            fragments2 = img_func.cut_matrix(expanded2, self.output_size, self.output_size)
            
            if len(fragments1) != len(fragments2):
                print(f"  Внимание: количество фрагментов не совпадает ({len(fragments1)} != {len(fragments2)})")
                min_len = min(len(fragments1), len(fragments2))
                fragments1 = fragments1[:min_len]
                fragments2 = fragments2[:min_len]
            
            total_loss = 0
            batch_count = 0
            
            for i in range(0, len(fragments1), self.batch_size):
                end_idx = min(i + self.batch_size, len(fragments1))
                batch_input = fragments1[i:end_idx] / 255.0
                batch_target = fragments2[i:end_idx] / 255.0
                
                if batch_input.shape[1:] != (self.input_size, self.input_size, 3):
                    print(f"  Неправильный размер входного батча: {batch_input.shape}")
                    continue
                
                if batch_target.shape[1:] != (self.output_size, self.output_size, 3):
                    print(f"  Неправильный размер целевого батча: {batch_target.shape}")
                    continue
                
                predictions = self.forward(batch_input)
                
                loss = self.backward(batch_input, batch_target, predictions)
                
                total_loss += loss
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                return avg_loss, True
            else:
                return 0, False
                
        except Exception as e:
            print(f"  Ошибка при обработке изображения {low_img_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0, False
    
    def train(self, num_images=None, auto_save=True):

        low_images = sorted([f for f in os.listdir(self.low_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        high_images = sorted([f for f in os.listdir(self.high_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        if num_images:
            low_images = low_images[:num_images]
            high_images = high_images[:num_images]
        
        if len(low_images) != len(high_images):
            print(f"Внимание: количество изображений не совпадает!")
            print(f"Low: {len(low_images)}, High: {len(high_images)}")
            print("Будем использовать минимальное количество")
            min_len = min(len(low_images), len(high_images))
            low_images = low_images[:min_len]
            high_images = high_images[:min_len]
        
        total_epochs = self.num_epochs
        total_images = len(low_images)
        
        print(f"Начинаем обучение...")
        print(f"Эпох: {total_epochs}, Изображений: {total_images}")
        print(f"Архитектура: вход {self.input_size}x{self.input_size}x3 -> выход {self.output_size}x{self.output_size}x3")
        print(f"Свертка: {self.num_filters} фильтров, ядро {self.kernel_size}x{self.kernel_size}")
        print(f"Веса будут сохраняться в: {self.weights_file}")
        
        for epoch in range(total_epochs):
            if self.stop_training:
                print("Обучение остановлено по команде пользователя.")
                break
                
            print(f"\n=== Эпоха {epoch+1}/{total_epochs} ===")
            epoch_start_time = time.time()
            total_loss = 0
            processed_images = 0
            
            for img_idx in range(total_images):
                if self.stop_training:
                    print("Обучение остановлено по команде пользователя.")
                    break
                
                img_start_time = time.time()
                low_img_name = low_images[img_idx]
                high_img_name = high_images[img_idx]
                
                print(f"\nОбработка изображения {img_idx+1}/{total_images}: {low_img_name}")
                
                img_loss, success = self.process_single_image(low_img_name, high_img_name)
                
                if success:
                    total_loss += img_loss
                    processed_images += 1
                    
                    img_time = time.time() - img_start_time
                    print(f"  Loss: {img_loss:.6f}, Время: {img_time:.1f}с")
                    
                    if auto_save:
                        self.save_weights()
                        print(f"  Веса сохранены в {self.weights_file}")
                    
                    if img_idx < total_images - 1 and not self.stop_training:
                        if self.wait_for_stop_command(self.wait_time):
                            print("Обучение остановлено по команде пользователя.")
                            break
            
            if processed_images > 0:
                avg_loss = total_loss / processed_images
                epoch_time = time.time() - epoch_start_time
                print(f"\nЭпоха {epoch+1} завершена. Средний Loss: {avg_loss:.6f}, Время: {epoch_time:.1f}с")
            
            if auto_save and not self.stop_training:
                backup_file = f"cnn_weights_epoch_{epoch+1}.npz"
                self.save_weights(backup_file)
                print(f"Резервная копия весов сохранена в: {backup_file}")
        
        if not self.stop_training:
            print("\nОбучение завершено!")
            if auto_save:
                print(f"Финальные веса сохранены в: {self.weights_file}")
    
    def enhance_image(self, image_path, save_path=None):

        print(f"Улучшение изображения: {image_path}")
        
        img = Image.open(image_path).convert('RGB')
        original_matrix = img_func.get_matrix_img(img)
        
        original_height, original_width, _ = original_matrix.shape
        
        print(f"Оригинальный размер: {original_width}x{original_height}")
        
        expanded = img_func.expand_matrix(original_matrix, self.input_size, self.input_size)
        fragments = img_func.cut_matrix(expanded, self.input_size, self.input_size)
        
        print(f"Создано {len(fragments)} фрагментов")
        
        enhanced_fragments = []
        
        for i in range(0, len(fragments), self.batch_size):
            end_idx = min(i + self.batch_size, len(fragments))
            batch = fragments[i:end_idx] / 255.0
            enhanced_batch = self.forward(batch)
            enhanced_fragments.append(enhanced_batch * 255)
            
            if i % (self.batch_size * 10) == 0:
                print(f"  Обработано {end_idx}/{len(fragments)} фрагментов")
        
        if enhanced_fragments:
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
                print(f"Улучшенное изображение сохранено в: {save_path}")
            
            print(f"Улучшение завершено. Новый размер: {enhanced_matrix.shape[1]}x{enhanced_matrix.shape[0]}")
            
            return enhanced_matrix
        else:
            print("Не удалось обработать изображение")
            return None