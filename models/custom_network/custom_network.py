# custom_network.py
import yaml
import requests
from zipfile import ZipFile
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import os
import random
import shutil
import csv
import subprocess
import importlib
from ultralytics import YOLOv10
from torchvision.models import get_model_weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
import json
import sys


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, class_to_idx=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.confusion_matrices = [] # to record training performance

        # 过滤掉隐藏文件夹
        classes = sorted([
            cls for cls in os.listdir(data_dir)
            if not cls.startswith('.') and os.path.isdir(os.path.join(data_dir, cls))
        ])

        print(f"CustomDataset: Classes found: {classes}")  # 调试输出

        for cls_name in classes:
            cls_path = os.path.join(data_dir, cls_name)
            #print(f"CustomDataset: Processing class: {cls_name}")  # 调试输出

            # 过滤掉隐藏文件
            img_names = sorted([
                img for img in os.listdir(cls_path)
                if not img.startswith('.') and os.path.isfile(os.path.join(cls_path, img))
            ])
            #print(f"CustomDataset: Found {len(img_names)} images for class '{cls_name}'")  # 调试输出

            for img_name in img_names:
                img_path = os.path.join(cls_path, img_name)
                self.data.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
class CustomNetwork:
    def __init__(self, mode, detector_cfg, classifier_cfg, training_cfg=None, prediction_cfg=None,
                 data_source_cfg=None):
        self.mode = mode
        self.device = 'cpu'

        # 初始化类别映射
        self.class_to_idx = {}
        self.idx_to_class = {}

        # 初始化数据源
        self.data_source_cfg = data_source_cfg
        self._init_data_source()

        # 根据模式和配置文件设置设备
        if self.mode == 'train' and training_cfg is not None:
            self.training_cfg = training_cfg
            self.device = training_cfg.get('device', 'cpu')
            print(f"Training on device: {self.device}")
        elif self.mode == 'predict' and prediction_cfg is not None:
            self.prediction_cfg = prediction_cfg
            self.device = prediction_cfg.get('device', 'cpu')
            print(f"Prediction on device: {self.device}")
        else:
            raise ValueError("Invalid mode or missing configuration.")

        # 存储分类器配置
        self.classifier_cfg = classifier_cfg

        # 初始化检测器
        self._init_detector(detector_cfg)

        # 初始化动态分类器
        self._init_classifier_dynamic()

        # 初始化处理器
        self._init_processor()
        # 初始化记录列表
        self.confusion_matrices = []
        self.accuracies = []
        self.f1_scores = []
        # 将模型移动到指定设备
        self.classifier.to(self.device)
        if self.mode == 'predict':
            self.detector.to(self.device)

    def _init_detector(self, detector_cfg):
        model_name = detector_cfg.get('model_name', 'yolov5s')
        pretrained = detector_cfg.get('pretrained', True)
        model_path = detector_cfg.get('model_path', None)
        
        self.expected_type_id = detector_cfg.get('expected_type_id', '80')
        if model_name.startswith('yolov10'):
            self.detector = YOLOv10(model_path,verbose=False)
        elif model_name.startswith('yolov5'):
            self.detector = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained,verbose=False)
        
        # to_device
        self.detector.to(self.device)

    def _init_classifier_dynamic(self):
        if self.mode != 'train':
            # 在预测模式下，加载类别映射
            class_mapping_path = self.classifier_cfg.get('class_mapping_path', 'models/resnet18/class_mapping.json')
            if not os.path.exists(class_mapping_path):
                raise ValueError(
                    "class_to_idx is not initialized. Ensure that the training was done or the class mappings are loaded.")
            with open(class_mapping_path, 'r') as f:
                self.class_to_idx = json.load(f)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            num_classes = len(self.class_to_idx)
            print(f"Loaded {num_classes} classes: {self.class_to_idx}")
        else:
            # 读取训练数据以确定类别数，排除 .DS_Store 等隐藏文件
            train_dir = os.path.join(self.training_cfg['data_dir'], 'train')
            classes = sorted(
                [cls for cls in os.listdir(train_dir) if not cls.startswith('.')]
            )
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            num_classes = len(classes)
            print(f"Detected {num_classes} classes: {self.class_to_idx}")

        # 初始化分类器
        model_type = self.classifier_cfg.get('model_type', 'torchvision')  # 'torchvision' 或 'custom'
        model_name = self.classifier_cfg.get('model_name', 'resnet18')

        if model_type == 'torchvision':
            pretrained = self.classifier_cfg.get('pretrained', False)
            model_path = self.classifier_cfg.get('model_path', None)
            weights = get_model_weights(model_name) if pretrained else None

            self.classifier = getattr(models, model_name)(weights=weights)
            if not pretrained:
                print(f"Initializing {model_name} without pretrained weights.")

            # 自动修改最后一层以适应分类任务
            self._modify_last_layer(model_name, num_classes)

            # 如果提供了 model_path，则加载模型权重
            if model_path and os.path.exists(model_path):
                try:
                    self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"Loaded classifier weights from {model_path}")
                except Exception as e:
                    print(f"Failed to load model weights from {model_path}: {e}")
            else:
                print(f"No model weights found at {model_path}. Using randomly initialized weights.")
        elif model_type == 'custom':
            # 加载自定义模型
            model_module = self.classifier_cfg.get('model_module', None)
            class_name = self.classifier_cfg.get('class_name', None)

            if not model_module or not class_name:
                raise ValueError("For custom models, 'model_module' and 'class_name' must be specified in the config.")

            try:
                custom_module = importlib.import_module(model_module)
                CustomModelClass = getattr(custom_module, class_name)
                # 假设自定义模型的初始化参数为 in_channels 和 num_classes
                in_channels = 3  # 默认输入通道数，可以根据需要调整或在配置中指定
                self.classifier = CustomModelClass(in_channels=in_channels, num_classes=num_classes)
                print(f"Initialized custom model '{class_name}' from module '{model_module}'")
            except ImportError as e:
                raise ImportError(f"Failed to import module '{model_module}': {e}")
            except AttributeError:
                raise AttributeError(f"Module '{model_module}' does not have a class '{class_name}'")
            except Exception as e:
                raise Exception(f"Failed to initialize custom model '{class_name}': {e}")

            # 自动修改自定义模型的最后一层
            self._modify_last_layer_custom(class_name, num_classes)

            # 如果提供了 model_path，则加载模型权重
            model_path = self.classifier_cfg.get('model_path', None)
            if model_path and os.path.exists(model_path):
                try:
                    self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"Loaded custom classifier weights from {model_path}")
                except Exception as e:
                    print(f"Failed to load custom model weights from {model_path}: {e}")
            else:
                print(f"No custom model weights found at {model_path}. Using randomly initialized weights.")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Must be 'torchvision' or 'custom'.")

        # 将分类器移动到指定设备
        self.classifier.to(self.device)

        # 打印 class_to_idx 映射
        print("Class to Index Mapping:")
        for class_name, index in self.class_to_idx.items():
            print(f"{class_name}: {index}")

        # 如果是训练模式，保存类别映射
        if self.mode == 'train':
            class_mapping_path = self.classifier_cfg.get('class_mapping_path', 'models/resnet18/class_mapping.json')
            os.makedirs(os.path.dirname(class_mapping_path), exist_ok=True)
            with open(class_mapping_path, 'w') as f:
                json.dump(self.class_to_idx, f)
            print(f"Saved class mappings to {class_mapping_path}")

    def _modify_last_layer_custom(self, class_name, num_classes):
        """
        自动修改自定义模型的最后一层以适应分类任务。
        假设自定义模型的最后一层名称为 'classifier' 或 'fc'。
        """
        try:
            if hasattr(self.classifier, 'classifier'):
                # 适用于类似 ResNet9 的模型
                in_features = self.classifier.classifier[-1].in_features
                self.classifier.classifier[-1] = nn.Linear(in_features, num_classes)
                print(f"Modified 'classifier' layer of custom model '{class_name}' to output {num_classes} classes.")
            elif hasattr(self.classifier, 'fc'):
                # 适用于其他可能的模型
                in_features = self.classifier.fc.in_features
                self.classifier.fc = nn.Linear(in_features, num_classes)
                print(f"Modified 'fc' layer of custom model '{class_name}' to output {num_classes} classes.")
            else:
                raise AttributeError(f"Custom model '{class_name}' does not have a 'classifier' or 'fc' attribute.")
        except Exception as e:
            raise Exception(f"Failed to modify the last layer of custom model '{class_name}': {e}")

    def _modify_last_layer(self, model_name, num_classes):
        if model_name.startswith('resnet'):
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
            print(f"Modified 'fc' layer of {model_name} to output {num_classes} classes.")
        elif model_name.startswith('vgg') or model_name.startswith('alexnet'):
            self.classifier.classifier[6] = nn.Linear(self.classifier.classifier[6].in_features, num_classes)
            print(f"Modified 'classifier[6]' layer of {model_name} to output {num_classes} classes.")
        elif model_name.startswith('mobilenet'):
            self.classifier.classifier[1] = nn.Linear(self.classifier.classifier[1].in_features, num_classes)
            print(f"Modified 'classifier[1]' layer of {model_name} to output {num_classes} classes.")
        else:
            raise ValueError(
                f"Model architecture for {model_name} is not supported. Please update the _modify_last_layer method.")
    def _init_processor(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # 如果需要，添加归一化
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    def _init_data_source(self):
        data_source = self.data_source_cfg
        data_type = data_source.get('type', 'local')

        if data_type == 'online':
            package = data_source.get('package')
            dataset_name = data_source.get('dataset_name')
            download = data_source.get('download', False)

            if download:
                # 安装指定的pip包
                print(f"Installing package: {package}")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"Package {package} installed successfully.")

            # 动态导入数据集模块
            try:
                dataset_module = importlib.import_module(dataset_name)
                self.dataset = dataset_module.get_dataset()  # 假设数据集模块有一个 get_dataset 方法
                self.training_cfg['data_dir'] = self.dataset.get_data_dir()
                print(f"Online dataset {dataset_name} loaded successfully.")
            except ImportError as e:
                raise ImportError(f"Failed to import dataset module {dataset_name}: {e}")
            except AttributeError:
                raise AttributeError(f"The dataset module {dataset_name} must have a 'get_dataset' method.")
        elif data_type == 'local':
            # 使用本地数据，路径已在训练配置中指定
            print("Using local dataset.")
        else:
            raise ValueError("data_source.type must be 'local' or 'online'.")

    def split_dataset(self, data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        print(f"Splitting dataset. Data directory: {data_dir}")

        # 获取所有非隐藏且为目录的类名
        classes = [
            cls for cls in os.listdir(data_dir)
            if not cls.startswith('.') and os.path.isdir(os.path.join(data_dir, cls))
        ]

        print(f"Classes found: {classes}")  # 调试输出

        for cls in classes:
            cls_path = os.path.join(data_dir, cls)

            # 获取所有非隐藏且为文件的图像文件名
            images = [
                img for img in os.listdir(cls_path)
                if not img.startswith('.') and os.path.isfile(os.path.join(cls_path, img))
            ]

            print(f"Found {len(images)} images for class '{cls}'")  # 调试输出

            random.shuffle(images)
            total = len(images)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)

            splits = {
                'train': images[:train_end],
                'val': images[train_end:val_end],
                'test': images[val_end:]
            }

            for split in splits:
                split_dir = os.path.join(output_dir, split, cls)
                os.makedirs(split_dir, exist_ok=True)
                for img_name in splits[split]:
                    src = os.path.join(cls_path, img_name)
                    dst = os.path.join(split_dir, img_name)
                    shutil.copy(src, dst)

        print("Dataset split successfully.")

    def process_detections(self, image, detections):
        cropped_images = []
        
        
        # Iterate over the detections (now using the 'data' field)
        for detection_all in detections:
            detection = detection_all.data
            if detection_all.id != self.expected_type_id:
                continue
            # Unpack detection values (x_min, y_min, x_max, y_max, conf, cls)
            x_min, y_min, x_max, y_max, conf, cls = detection
            
            # Crop the image using the bounding box
            cropped_img = image.crop((x_min, y_min, x_max, y_max))
            
            # Apply the transformation
            cropped_img = self.transform(cropped_img)
            
            # Append the processed image to the list
            cropped_images.append(cropped_img)
        
        # If there are cropped images, stack them into a tensor, otherwise return None
        if cropped_images:
            return torch.stack(cropped_images)
        else:
            return None

    def train(self):
        # 获取训练配置
        base_data_dir = self.training_cfg.get('data_dir')  # 基础数据目录，应该是 'data'
        data_dir = os.path.join(base_data_dir, 'train')  # 实际包含类别子目录的训练数据目录
        output_dir = self.training_cfg.get('temp_dir', 'data/temp')

        print(f"Training with train_data_dir: {data_dir}, output_dir: {output_dir}")  # 调试输出

        # 数据集划分
        if not os.path.exists(output_dir):
            self.split_dataset(data_dir, output_dir)
            print("Dataset split successfully.")
        else:
            shutil.rmtree(output_dir)  # 递归删除非空目录
            self.split_dataset(data_dir, output_dir)
            print("Dataset split successfully.")

        # 定义数据集和数据加载器
        train_loader, val_loader = self._get_data_loaders(output_dir, self.training_cfg.get('batch_size', 32))

        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 动态选择优化器
        optimizer_cfg = self.training_cfg.get('optimizer', {})
        optimizer_type = optimizer_cfg.get('type', 'SGD')
        lr = optimizer_cfg.get('lr', 0.001)

        if optimizer_type == 'SGD':
            momentum = optimizer_cfg.get('momentum', 0.9)
            optimizer = optim.SGD(
                self.classifier.parameters(),
                lr=lr,
                momentum=momentum
            )
        elif optimizer_type == 'Adam':
            weight_decay = optimizer_cfg.get('weight_decay', 0.0)
            optimizer = optim.Adam(
                self.classifier.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'RMSprop':
            momentum = optimizer_cfg.get('momentum', 0.9)
            weight_decay = optimizer_cfg.get('weight_decay', 0.0)
            optimizer = optim.RMSprop(
                self.classifier.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        print(f"Using optimizer: {optimizer_type} with parameters: {optimizer_cfg}")  # 调试输出

        num_epochs = self.training_cfg.get('num_epochs', 10)

        best_f1 = 0.0
        best_model_path = self.training_cfg.get('best_model_path', 'models/resnet18/best_classifier.pth')

        # 训练循环
        for epoch in range(num_epochs):
            self.classifier.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                # 将数据迁移到设备
                inputs = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.classifier(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 计算并打印进度
                if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            # Validation step
            self.classifier.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    inputs = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.classifier(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            f1 = f1_score(all_labels, all_preds, average='weighted')  # 计算加权 F1 分数
            f1percent=f1*100
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, F1 Score: {f1percent:.2f}%')

            # 计算混淆矩阵
            cm = confusion_matrix(all_labels, all_preds)
            self.confusion_matrices.append(cm.tolist())  # 转换为列表以便JSON序列化
            self.accuracies.append(val_accuracy)
            self.f1_scores.append(f1)
            print(f'Confusion Matrix for Epoch {epoch + 1}:')
            print(cm)

            # 保存模型
            save_path = self.training_cfg.get('save_path', 'classifier.pth')
            torch.save(self.classifier.state_dict(), save_path)
            print(f"Model saved to {save_path}\n")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.classifier.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path} with F1 Score: {best_f1:.2f}")

            # 保存所有混淆矩阵和指标到JSON文件
            metrics_path = self.training_cfg.get('metrics_path', 'training/metrics.json')
            metrics = {
                'confusion_matrices': self.confusion_matrices,
                'accuracies': self.accuracies,
                'f1_scores': self.f1_scores
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            print(f"Saved all metrics to {metrics_path}")

        print("Training complete!")

    def predict(self):
        import csv  # 确保在predict方法内导入csv模块

        # 设置分类器为评估模式
        self.classifier.eval()

        # 获取预测配置
        predict_dir = self.prediction_cfg.get('predict_dir')
        output_dir = self.prediction_cfg.get('output_dir')
        os.makedirs(output_dir, exist_ok=True)

        # 获取是否使用检测器的配置
        is_detector = self.prediction_cfg.get('is_detector', 'false').lower() == 'true'

        image_names = os.listdir(predict_dir)

        # 列表用于存储所有预测结果
        predictions_list = []

        for img_name in image_names:
            img_path = os.path.join(predict_dir, img_name)
            image = Image.open(img_path).convert('RGB')

            if is_detector:
                # 执行检测
                results = self.detector(image)

                # 确保 results 是列表且不为空
                if isinstance(results, list) and len(results) > 0:
                    # 正确访问 detections
                    detections = results[0].boxes  # 根据YOLOv10的输出调整
                    print(f"Detections for {img_name}: {detections}")
                else:
                    detections = []
                    print(f"No detections for {img_name}")

                # 数据处理
                inputs = self.process_detections(image, detections)
                if inputs is None:
                    print(f"No valid detections to classify for {img_name}")
                    continue
            else:
                # 不使用检测器，直接使用整个图像
                inputs = self.transform(image).unsqueeze(0).to(self.device)
                print(f"Using entire image for classification: {img_name}")

            # 预测
            outputs = self.classifier(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            for i in range(outputs.size(0)):
                prob, pred = probabilities[i].max(0)
                class_num = pred.item()
                class_name = self.idx_to_class.get(class_num, "Unknown")
                probability = prob.item() * 100  # 转换为百分比

                # 将预测结果添加到列表中
                predictions_list.append({
                    'class_number': class_num,
                    'class_name': class_name,
                    'probability': f"{probability:.2f}%"
                })

        # 将所有预测结果写入CSV文件
        csv_path = os.path.join(output_dir, 'predictions.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['class_number', 'class_name', 'probability']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for pred in predictions_list:
                writer.writerow(pred)

        print(f"Saved all predictions to {csv_path}")

    def _get_data_loaders(self, data_dir, batch_size):
        # 使用顶层定义的 CustomDataset 类
        train_dataset = CustomDataset(
            os.path.join(data_dir, 'train'),
            transform=self.transform,
            class_to_idx=self.class_to_idx
        )
        val_dataset = CustomDataset(
            os.path.join(data_dir, 'val'),
            transform=self.transform,
            class_to_idx=self.class_to_idx
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        return train_loader, val_loader

    def _process_batch(self, images):
        inputs = []
        #print(len(images))
        for img_tensor in images:
            img = transforms.ToPILImage()(img_tensor)
            #print(self.detector.verbose)
            detections = []
            results = self.detector(img,verbose=False)
            if ((self.mode == 'train' and self.training_cfg.get('is_detector')=='true') or
                    (self.mode=='prediction' and self.prediction_cfg.get('is_detector')=='true')):
            # 使用 results.boxes 获取检测框
                detections = results[0].boxes
            
            if detections is not None and len(detections) > 0:
                detections = detections.cpu().numpy()  # 转换为 numpy 数组
                input_tensor = self.process_detections(img, detections)
                if input_tensor is not None and len(input_tensor) == 1:
                    inputs.append(input_tensor)
                else:
                    input_tensor = self.transform(img).unsqueeze(0)
                    inputs.append(input_tensor)
            else:
                # 如果没有检测到对象，将整个图像传入分类器
                input_tensor = self.transform(img).unsqueeze(0)  # 添加一个维度，确保 batch size 为 1
                inputs.append(input_tensor)
        
        if not inputs:
            return None
        
        # 合并所有图像的输入，形成一个大的 batch
        # 确保输入张量的维度一致
        inputs = torch.cat(inputs, dim=0)
        #print(f"Inputs shape: {inputs.shape}")
        
        # 返回输入时，确保返回的数据是一个批次
        return inputs.to(self.device)



    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'predict':
            self.predict()
        else:
            print("Invalid mode. Please choose 'train' or 'predict'.")
