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
        self.confusion_matrices = []  # to record training performance

        # Filter out hidden folders
        classes = sorted([
            cls for cls in os.listdir(data_dir)
            if not cls.startswith('.') and os.path.isdir(os.path.join(data_dir, cls))
        ])

        print(f"CustomDataset: Classes found: {classes}")  # Debug output

        for cls_name in classes:
            cls_path = os.path.join(data_dir, cls_name)
            # print(f"CustomDataset: Processing class: {cls_name}")  # Debug output

            # Filter out hidden files
            img_names = sorted([
                img for img in os.listdir(cls_path)
                if not img.startswith('.') and os.path.isfile(os.path.join(cls_path, img))
            ])
            # print(f"CustomDataset: Found {len(img_names)} images for class '{cls_name}'")  # Debug output

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

        # Initialize class mapping
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Initialize data source
        self.data_source_cfg = data_source_cfg
        self._init_data_source()

        # Set device based on mode and configuration
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

        # Store classifier configuration
        self.classifier_cfg = classifier_cfg

        # Initialize detector
        self._init_detector(detector_cfg)

        # Initialize dynamic classifier
        self._init_classifier_dynamic()

        # Initialize processor
        self._init_processor()
        # Initialize recording lists
        self.confusion_matrices = []
        self.accuracies = []
        self.f1_scores = []
        # Move model to specified device
        self.classifier.to(self.device)
        if self.mode == 'predict':
            self.detector.to(self.device)

    def _init_detector(self, detector_cfg):
        model_name = detector_cfg.get('model_name', 'yolov5s')
        pretrained = detector_cfg.get('pretrained', True)
        model_path = detector_cfg.get('model_path', None)

        self.expected_type_id = detector_cfg.get('expected_type_id', '80')
        if model_name.startswith('yolov10'):
            self.detector = YOLOv10(model_path, verbose=False)
        elif model_name.startswith('yolov5'):
            self.detector = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained, verbose=False)

        # Move to device
        self.detector.to(self.device)

    def _init_classifier_dynamic(self):
        if self.mode != 'train':
            # In prediction mode, load class mapping
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
            # Read training data to determine the number of classes, excluding hidden files like .DS_Store
            train_dir = os.path.join(self.training_cfg['data_dir'], 'train')
            classes = sorted(
                [cls for cls in os.listdir(train_dir) if not cls.startswith('.')]
            )
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            num_classes = len(classes)
            print(f"Detected {num_classes} classes: {self.class_to_idx}")

        # Initialize classifier
        model_type = self.classifier_cfg.get('model_type', 'torchvision')  # 'torchvision' or 'custom'
        model_name = self.classifier_cfg.get('model_name', 'resnet18')

        if model_type == 'torchvision':
            pretrained = self.classifier_cfg.get('pretrained', False)
            model_path = self.classifier_cfg.get('model_path', None)
            weights = get_model_weights(model_name) if pretrained else None

            self.classifier = getattr(models, model_name)(weights=weights)
            if not pretrained:
                print(f"Initializing {model_name} without pretrained weights.")

            # Automatically modify the last layer to fit the classification task
            self._modify_last_layer(model_name, num_classes)

            # If model_path is provided, load model weights
            if model_path and os.path.exists(model_path):
                try:
                    self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"Loaded classifier weights from {model_path}")
                except Exception as e:
                    print(f"Failed to load model weights from {model_path}: {e}")
            else:
                print(f"No model weights found at {model_path}. Using randomly initialized weights.")
        elif model_type == 'custom':
            # Load custom model
            model_module = self.classifier_cfg.get('model_module', None)
            class_name = self.classifier_cfg.get('class_name', None)

            if not model_module or not class_name:
                raise ValueError("For custom models, 'model_module' and 'class_name' must be specified in the config.")

            try:
                custom_module = importlib.import_module(model_module)
                CustomModelClass = getattr(custom_module, class_name)
                # Assume the custom model's initialization parameters are in_channels and num_classes
                in_channels = 3  # Default input channels, can be adjusted or specified in the config
                self.classifier = CustomModelClass(in_channels=in_channels, num_classes=num_classes)
                print(f"Initialized custom model '{class_name}' from module '{model_module}'")
            except ImportError as e:
                raise ImportError(f"Failed to import module '{model_module}': {e}")
            except AttributeError:
                raise AttributeError(f"Module '{model_module}' does not have a class '{class_name}'")
            except Exception as e:
                raise Exception(f"Failed to initialize custom model '{class_name}': {e}")

            # Automatically modify the last layer of the custom model
            self._modify_last_layer_custom(class_name, num_classes)

            # If model_path is provided, load model weights
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

        # Move classifier to specified device
        self.classifier.to(self.device)

        # Print class_to_idx mapping
        print("Class to Index Mapping:")
        for class_name, index in self.class_to_idx.items():
            print(f"{class_name}: {index}")

        # If in training mode, save class mapping
        if self.mode == 'train':
            class_mapping_path = self.classifier_cfg.get('class_mapping_path', 'models/resnet18/class_mapping.json')
            os.makedirs(os.path.dirname(class_mapping_path), exist_ok=True)
            with open(class_mapping_path, 'w') as f:
                json.dump(self.class_to_idx, f)
            print(f"Saved class mappings to {class_mapping_path}")

    def _modify_last_layer_custom(self, class_name, num_classes):
        """
        Automatically modifies the last layer of the custom model to fit the classification task.
        Assumes that the custom model's last layer is named 'classifier' or 'fc'.
        """
        try:
            if hasattr(self.classifier, 'classifier'):
                # Suitable for models like ResNet9
                in_features = self.classifier.classifier[-1].in_features
                self.classifier.classifier[-1] = nn.Linear(in_features, num_classes)
                print(f"Modified 'classifier' layer of custom model '{class_name}' to output {num_classes} classes.")
            elif hasattr(self.classifier, 'fc'):
                # Suitable for other potential models
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
            # Add normalization if needed
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
                # Install specified pip package
                print(f"Installing package: {package}")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"Package {package} installed successfully.")

            # Dynamically import dataset module
            try:
                dataset_module = importlib.import_module(dataset_name)
                self.dataset = dataset_module.get_dataset()  # Assumes the dataset module has a get_dataset method
                self.training_cfg['data_dir'] = self.dataset.get_data_dir()
                print(f"Online dataset {dataset_name} loaded successfully.")
            except ImportError as e:
                raise ImportError(f"Failed to import dataset module {dataset_name}: {e}")
            except AttributeError:
                raise AttributeError(f"The dataset module {dataset_name} must have a 'get_dataset' method.")
        elif data_type == 'local':
            # Use local data, path is already specified in training configuration
            print("Using local dataset.")
        else:
            raise ValueError("data_source.type must be 'local' or 'online'.")

    def split_dataset(self, data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        print(f"Splitting dataset. Data directory: {data_dir}")

        # Get all class names that are not hidden and are directories
        classes = [
            cls for cls in os.listdir(data_dir)
            if not cls.startswith('.') and os.path.isdir(os.path.join(data_dir, cls))
        ]

        print(f"Classes found: {classes}")  # Debug output

        for cls in classes:
            cls_path = os.path.join(data_dir, cls)

            # Get all image filenames that are not hidden and are files
            images = [
                img for img in os.listdir(cls_path)
                if not img.startswith('.') and os.path.isfile(os.path.join(cls_path, img))
            ]

            print(f"Found {len(images)} images for class '{cls}'")  # Debug output

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
        # Get training configuration
        base_data_dir = self.training_cfg.get('data_dir')  # Base data directory, should be 'data'
        data_dir = os.path.join(base_data_dir, 'train')  # Actual training data directory containing class subdirectories
        output_dir = self.training_cfg.get('temp_dir', 'data/temp')

        print(f"Training with train_data_dir: {data_dir}, output_dir: {output_dir}")  # Debug output

        # Split dataset
        if not os.path.exists(output_dir):
            self.split_dataset(data_dir, output_dir)
            print("Dataset split successfully.")
        else:
            shutil.rmtree(output_dir)  # Recursively delete non-empty directory
            self.split_dataset(data_dir, output_dir)
            print("Dataset split successfully.")

        # Define datasets and data loaders
        train_loader, val_loader = self._get_data_loaders(output_dir, self.training_cfg.get('batch_size', 32))

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Dynamically select optimizer
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

        print(f"Using optimizer: {optimizer_type} with parameters: {optimizer_cfg}")  # Debug output

        num_epochs = self.training_cfg.get('num_epochs', 10)

        best_f1 = 0.0
        best_model_path = self.training_cfg.get('best_model_path', 'models/resnet18/best_classifier.pth')

        # Training loop
        for epoch in range(num_epochs):
            self.classifier.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Move data to device
                inputs = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.classifier(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculate and print progress
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
            f1 = f1_score(all_labels, all_preds, average='weighted')  # Calculate weighted F1 score
            f1percent = f1 * 100
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, F1 Score: {f1percent:.2f}%')

            # Compute confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            self.confusion_matrices.append(cm.tolist())  # Convert to list for JSON serialization
            self.accuracies.append(val_accuracy)
            self.f1_scores.append(f1)
            print(f'Confusion Matrix for Epoch {epoch + 1}:')
            print(cm)

            # Save model
            save_path = self.training_cfg.get('save_path', 'classifier.pth')
            torch.save(self.classifier.state_dict(), save_path)
            print(f"Model saved to {save_path}\n")
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.classifier.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path} with F1 Score: {best_f1:.2f}")

            # Save all confusion matrices and metrics to JSON file
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
        import csv  # Ensure csv module is imported within predict method

        # Set classifier to evaluation mode
        self.classifier.eval()

        # Get prediction configuration
        predict_dir = self.prediction_cfg.get('predict_dir')
        output_dir = self.prediction_cfg.get('output_dir')
        os.makedirs(output_dir, exist_ok=True)

        # Get whether to use detector from configuration
        is_detector = self.prediction_cfg.get('is_detector', 'false').lower() == 'true'

        image_names = os.listdir(predict_dir)

        # List to store all prediction results
        predictions_list = []

        for img_name in image_names:
            img_path = os.path.join(predict_dir, img_name)
            image = Image.open(img_path).convert('RGB')

            if is_detector:
                # Perform detection
                results = self.detector(image)

                # Ensure results is a list and not empty
                if isinstance(results, list) and len(results) > 0:
                    # Correctly access detections
                    detections = results[0].boxes  # Adjust based on YOLOv10's output
                    print(f"Detections for {img_name}: {detections}")
                else:
                    detections = []
                    print(f"No detections for {img_name}")

                # Data processing
                inputs = self.process_detections(image, detections)
                if inputs is None:
                    print(f"No valid detections to classify for {img_name}")
                    continue
            else:
                # Do not use detector, use the entire image
                inputs = self.transform(image).unsqueeze(0).to(self.device)
                print(f"Using entire image for classification: {img_name}")

            # Prediction
            outputs = self.classifier(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            for i in range(outputs.size(0)):
                prob, pred = probabilities[i].max(0)
                class_num = pred.item()
                class_name = self.idx_to_class.get(class_num, "Unknown")
                probability = prob.item() * 100  # Convert to percentage

                # Add prediction result to list
                predictions_list.append({
                    'class_number': class_num,
                    'class_name': class_name,
                    'probability': f"{probability:.2f}%"
                })

        # Write all prediction results to CSV file
        csv_path = os.path.join(output_dir, 'predictions.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['class_number', 'class_name', 'probability']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for pred in predictions_list:
                writer.writerow(pred)

        print(f"Saved all predictions to {csv_path}")

    def _get_data_loaders(self, data_dir, batch_size):
        # Use the CustomDataset class defined above
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
        # print(len(images))
        for img_tensor in images:
            img = transforms.ToPILImage()(img_tensor)
            # print(self.detector.verbose)
            detections = []
            results = self.detector(img, verbose=False)
            if ((self.mode == 'train' and self.training_cfg.get('is_detector') == 'true') or
                    (self.mode == 'prediction' and self.prediction_cfg.get('is_detector') == 'true')):
                # Use results.boxes to get bounding boxes
                detections = results[0].boxes

            if detections is not None and len(detections) > 0:
                detections = detections.cpu().numpy()  # Convert to numpy array
                input_tensor = self.process_detections(img, detections)
                if input_tensor is not None and len(input_tensor) == 1:
                    inputs.append(input_tensor)
                else:
                    input_tensor = self.transform(img).unsqueeze(0)
                    inputs.append(input_tensor)
            else:
                # If no objects detected, pass the entire image to the classifier
                input_tensor = self.transform(img).unsqueeze(0)  # Add a dimension to ensure batch size of 1
                inputs.append(input_tensor)

        if not inputs:
            return None

        # Combine all image inputs to form a large batch
        # Ensure input tensors have consistent dimensions
        inputs = torch.cat(inputs, dim=0)
        # print(f"Inputs shape: {inputs.shape}")

        # Return input ensuring data is a single batch
        return inputs.to(self.device)

    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'predict':
            self.predict()
        else:
            print("Invalid mode. Please choose 'train' or 'predict'.")