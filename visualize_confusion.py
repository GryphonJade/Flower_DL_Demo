# visualize_confusion.py

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_metrics(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def load_class_names(class_mapping_path):
    with open(class_mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    # 根据索引排序类别名称
    class_names = [None] * len(class_to_idx)
    for cls, idx in class_to_idx.items():
        class_names[idx] = cls
    return class_names


def plot_confusion_matrix(cm, epoch, class_names, output_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_epoch_{epoch}.png'))
    plt.close()


def plot_metrics(accuracies, f1_scores, output_dir):
    epochs = range(1, len(accuracies) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制准确率
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracies, 'bo-', label='Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # 绘制 F1 分数
    plt.subplot(1, 2, 2)
    plt.plot(epochs, f1_scores, 'ro-', label='F1 Score')
    plt.title('Validation F1 Score per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'accuracy_f1_scores.png'))
    plt.close()


def main():
    # 读取配置文件
    config_path = 'configs/config.yaml'  # 替换为您的 config.yaml 路径
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return

    config = load_config(config_path)

    # 提取训练相关的配置
    training_cfg = config.get('training', {})
    metrics_path = training_cfg.get('metrics_path', 'training/metrics.json')
    class_mapping_path = training_cfg.get('class_mapping_path', 'models/resnet18/class_mapping.json')

    # 设置可视化输出目录
    confusion_output_dir = training_cfg.get('confusion_output_dir', 'visualizations/confusion_matrices')
    metrics_output_dir = training_cfg.get('metrics_output_dir', 'visualizations/metrics')

    # 检查路径是否存在
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}")
        return

    if not os.path.exists(class_mapping_path):
        print(f"Class mapping file not found at {class_mapping_path}")
        return

    # 加载指标和类别名称
    metrics = load_metrics(metrics_path)
    confusion_matrices = metrics.get('confusion_matrices', [])
    accuracies = metrics.get('accuracies', [])
    f1_scores = metrics.get('f1_scores', [])


    class_names = load_class_names(class_mapping_path)

    num_epochs = len(confusion_matrices)

    for epoch in range(1, num_epochs + 1):
        cm = np.array(confusion_matrices[epoch - 1])
        plot_confusion_matrix(cm, epoch, class_names, confusion_output_dir)
        print(f"Saved confusion matrix for Epoch {epoch} to {confusion_output_dir}")

    # 绘制准确率和F1分数的变化曲线
    plot_metrics(accuracies, f1_scores, metrics_output_dir)
    print(f"Saved accuracy and F1 score plots to {metrics_output_dir}")

    print("All visualizations have been generated.")


if __name__ == '__main__':
    main()