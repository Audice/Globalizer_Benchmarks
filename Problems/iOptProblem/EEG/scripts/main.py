import torch
import FuzzyPart as fp

def example_usage():
    # Создаем тестовые данные
    x = torch.randn(4, 23, 2560)

    # 1. Simple CNN
    print("\n1. Simple 1D CNN:")
    config = fp.FuzzyCNNConfig(
        backbone_name='simple',
        input_channels=23,
        input_width=2560,
        backbone_feature_dim=512,
        cnn_embedding_dim=64,
        num_rules=16,
        backbone_dropout=0.3,
        extra_params={}
    )
    model = fp.CNNWithFuzzy(config)
    output, rules, embedding = model(x)
    print(f"Выходной shape: {output.shape}")
    print(f"Embedding shape: {embedding.shape}")

    # 2. Deep CNN
    print("\n2. Deep 1D CNN:")
    config.backbone_name = 'deep'
    config.extra_params = {}
    model = fp.CNNWithFuzzy(config)
    output, rules, embedding = model(x)
    print(f"   Output shape: {output.shape}")

    # 3. ResNet18
    print("\n3. ResNet18 1D:")
    config.backbone_name = 'resnet18'
    config.extra_params = {}
    model = fp.CNNWithFuzzy(config)
    output, rules, embedding = model(x)
    print(f"   Output shape: {output.shape}")

    # 4. ResNet34
    print("\n4. ResNet34 1D:")
    config.backbone_name = 'resnet34'
    config.extra_params = {}
    model = fp.CNNWithFuzzy(config)
    output, rules, embedding = model(x)
    print(f"   Output shape: {output.shape}")

    # 5. Inception
    print("\n5. Inception 1D:")
    config.backbone_name = 'inception'
    config.extra_params = {}
    model = fp.CNNWithFuzzy(config)
    output, rules, embedding = model(x)
    print(f"   Output shape: {output.shape}")

    # 6. DenseNet
    print("\n6. DenseNet 1D:")
    config.backbone_name = 'densenet'
    config.extra_params = {'growth_rate': 32, 'n_layers': 4}
    model = fp.CNNWithFuzzy(config)
    output, rules, embedding = model(x)
    print(f"Output shape: {output.shape}")

    # 7. Сравнение всех моделей
    models_to_test = [
        ('simple', {}),
        ('deep', {}),
        ('resnet18', {}),
        ('resnet34', {}),
        ('inception', {}),
        ('densenet', {'growth_rate': 32, 'n_layers': 4}),
    ]

    for model_name, extra_params in models_to_test:
        config.backbone_name = model_name
        config.extra_params = extra_params

        model = fp.CNNWithFuzzy(config)
        info = model.get_backbone_info()
        print(f"\n{model_name.upper()}:")
        print(f"   Feature dim: {info['feature_dim']}")
        print(f"   Parameters: {info['total_parameters']:,}")

        # Тестовый forward
        output, rules, embedding = model(x)
        print(f"Выход : {output.shape}")

if __name__ == "__main__":
    # ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ 1D CNN АРХИТЕКТУР
    example_usage()