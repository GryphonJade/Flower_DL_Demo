# auto_update_config.py

import yaml
import argparse
import os
import sys

def update_config(config_path):
    """
    自动化更新 config.yaml 中的 classifier 和 training 部分。
    根据 model_type 和 model_name 设置相关路径和参数，并创建必要的目录。
    """
    if not os.path.exists(config_path):
        print(f"错误: 配置文件 '{config_path}' 不存在。")
        sys.exit(1)

    # 读取现有的 config.yaml
    with open(config_path, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"错误: 读取配置文件时出错: {exc}")
            sys.exit(1)

    classifier_cfg = config.get('classifier', {})
    training_cfg = config.get('training', {})

    # 获取当前的 model_type 和 model_name
    model_type = classifier_cfg.get('model_type', 'torchvision').lower()
    model_name = classifier_cfg.get('model_name', 'resnet34')

    print(f"当前 model_type: '{model_type}'")
    print(f"当前 model_name: '{model_name}'")

    # 根据 model_type 设置相关字段
    if model_type == 'custom':
        # 确保 'class_name' 和 'model_module' 已设置
        class_name = classifier_cfg.get('class_name', None)
        model_module = classifier_cfg.get('model_module', None)

        if not class_name or not model_module:
            print("错误: 对于自定义模型，'class_name' 和 'model_module' 必须在配置文件中指定。")
            sys.exit(1)

        # 设置 paths based on model_name
        classifier_cfg['model_path'] = f'models/{model_name}/classifier.pth'
        classifier_cfg['class_mapping_path'] = f'models/{model_name}/class_mapping.json'

        training_cfg['save_path'] = f'models/{model_name}/classifier.pth'
        training_cfg['best_model_path'] = f'models/{model_name}/f1best.pth'

        training_cfg['confusion_path'] = f'training/{model_name}/confusion_matrices.json'
        training_cfg['metrics_path'] = f'training/{model_name}/metrics.json'

        training_cfg['confusion_output_dir'] = f'visualizations/{model_name}/confusion_matrices'
        training_cfg['metrics_output_dir'] = f'visualizations/{model_name}/metrics'

        print(f"设置 'classifier.model_path' 为: {classifier_cfg['model_path']}")
        print(f"设置 'classifier.class_mapping_path' 为: {classifier_cfg['class_mapping_path']}")
        print(f"设置 'training.save_path' 为: {training_cfg['save_path']}")
        print(f"设置 'training.best_model_path' 为: {training_cfg['best_model_path']}")
        print(f"设置 'training.confusion_path' 为: {training_cfg['confusion_path']}")
        print(f"设置 'training.metrics_path' 为: {training_cfg['metrics_path']}")
        print(f"设置 'training.confusion_output_dir' 为: {training_cfg['confusion_output_dir']}")
        print(f"设置 'training.metrics_output_dir' 为: {training_cfg['metrics_output_dir']}")

    elif model_type == 'torchvision':
        # 移除 'model_module' 和 'class_name' 字段如果存在
        removed_model_module = classifier_cfg.pop('model_module', None)
        removed_class_name = classifier_cfg.pop('class_name', None)
        if removed_model_module:
            print(f"移除 'classifier.model_module': {removed_model_module}")
        if removed_class_name:
            print(f"移除 'classifier.class_name': {removed_class_name}")

        # 设置 paths based on model_name
        classifier_cfg['model_path'] = f'models/{model_name}/classifier.pth'
        classifier_cfg['class_mapping_path'] = f'models/{model_name}/class_mapping.json'

        training_cfg['save_path'] = f'models/{model_name}/classifier.pth'
        training_cfg['best_model_path'] = f'models/{model_name}/f1best.pth'

        training_cfg['confusion_path'] = f'training/{model_name}/confusion_matrices.json'
        training_cfg['metrics_path'] = f'training/{model_name}/metrics.json'

        training_cfg['confusion_output_dir'] = f'visualizations/{model_name}/confusion_matrices'
        training_cfg['metrics_output_dir'] = f'visualizations/{model_name}/metrics'

        print(f"设置 'classifier.model_path' 为: {classifier_cfg['model_path']}")
        print(f"设置 'classifier.class_mapping_path' 为: {classifier_cfg['class_mapping_path']}")
        print(f"设置 'training.save_path' 为: {training_cfg['save_path']}")
        print(f"设置 'training.best_model_path' 为: {training_cfg['best_model_path']}")
        print(f"设置 'training.confusion_path' 为: {training_cfg['confusion_path']}")
        print(f"设置 'training.metrics_path' 为: {training_cfg['metrics_path']}")
        print(f"设置 'training.confusion_output_dir' 为: {training_cfg['confusion_output_dir']}")
        print(f"设置 'training.metrics_output_dir' 为: {training_cfg['metrics_output_dir']}")

    else:
        print(f"错误: 不支持的 model_type '{model_type}'。请选择 'torchvision' 或 'custom'。")
        sys.exit(1)

    # 更新 'model_type', 'model_name', 'pretrained'
    classifier_cfg['model_type'] = model_type
    classifier_cfg['model_name'] = model_name
    classifier_cfg['pretrained'] = classifier_cfg.get('pretrained', False)

    # 确保相关目录存在
    paths_to_create = [
        classifier_cfg.get('model_path'),
        classifier_cfg.get('class_mapping_path'),
        training_cfg.get('save_path'),
        training_cfg.get('best_model_path'),
        training_cfg.get('confusion_path'),
        training_cfg.get('metrics_path'),
        training_cfg.get('confusion_output_dir'),
        training_cfg.get('metrics_output_dir'),
    ]

    for path in paths_to_create:
        if path:
            dir_path = os.path.dirname(path)
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"创建目录: {dir_path}")
                except Exception as e:
                    print(f"错误: 创建目录 '{dir_path}' 时出错: {e}")
            else:
                print(f"目录已存在: {dir_path}")

    # 更新 config.yaml
    config['classifier'] = classifier_cfg
    config['training'] = training_cfg

    # 写回到 config.yaml
    try:
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, sort_keys=False, allow_unicode=True)
        print(f"已成功更新 '{config_path}' 文件。")
    except Exception as e:
        print(f"错误: 写回配置文件时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动化更新 config.yaml 配置文件")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='config.yaml 文件路径')

    args = parser.parse_args()

    update_config(args.config)