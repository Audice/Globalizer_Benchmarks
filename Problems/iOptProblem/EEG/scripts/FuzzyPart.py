import torch
import torch.nn as nn
import timm
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field
import warnings
from EEG.scripts.ModelArchitectures import *


class FuzzyTSKLayer(nn.Module):
    """
    Слой нечеткой логики Takagi-Sugeno-Kang с поддержкой TSK-0 и TSK-1
    """

    def __init__(self,
                 num_features: int,
                 num_rules: int = 16,
                 num_outputs: int = 1,
                 init_sigma: float = 1.0,
                 use_tsk1: bool = False,
                 dropout: float = 0.0):
        super(FuzzyTSKLayer, self).__init__()

        self.num_features = num_features
        self.num_rules = num_rules
        self.num_outputs = num_outputs
        self.use_tsk1 = use_tsk1

        # Функции принадлежности
        means_init = torch.linspace(-2, 2, num_rules).unsqueeze(1).repeat(1, num_features)
        means_init += torch.randn(num_rules, num_features) * 0.1
        self.means = nn.Parameter(means_init)
        self.sigmas = nn.Parameter(torch.ones(num_rules, num_features) * init_sigma)

        # Выход правил
        if use_tsk1:
            self.rule_weights = nn.Parameter(torch.randn(num_rules, num_outputs, num_features) * 0.01)
            self.rule_bias = nn.Parameter(torch.zeros(num_rules, num_outputs))
        else:
            self.rule_weights = nn.Parameter(torch.randn(num_rules, num_outputs) * 0.01)
            self.rule_bias = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.epsilon = 1e-6

        self._initialize_weights()

    def _initialize_weights(self):
        if self.use_tsk1:
            nn.init.xavier_uniform_(self.rule_weights)
            nn.init.zeros_(self.rule_bias)
        else:
            nn.init.xavier_uniform_(self.rule_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        x_expanded = x.unsqueeze(1)
        means_expanded = self.means.unsqueeze(0)
        sigmas_expanded = torch.abs(self.sigmas.unsqueeze(0)) + self.epsilon

        diff = x_expanded - means_expanded
        exponent = -0.5 * (diff / sigmas_expanded) ** 2
        rule_activations = torch.exp(exponent.sum(dim=2))
        rule_activations = self.dropout(rule_activations)

        sum_acts = rule_activations.sum(dim=1, keepdim=True) + self.epsilon
        normalized_activations = rule_activations / sum_acts

        if self.use_tsk1:
            rule_outputs = torch.einsum('r o f, b f -> b r o', self.rule_weights, x)
            rule_outputs = rule_outputs + self.rule_bias.unsqueeze(0)
            output = torch.einsum('b r, b r o -> b o', normalized_activations, rule_outputs)
        else:
            output = torch.mm(normalized_activations, self.rule_weights)

        return output, normalized_activations


@dataclass
class FuzzyCNNConfig:
    """Конфигурация для гиперпараметрической оптимизации"""

    # Параметры бэкбона
    backbone_name: str = 'simple'  # 'simple', 'deep', 'resnet18', 'resnet34', 'resnet50', 'inception', 'densenet'
    input_channels: int = 23
    input_width: int = 2560
    backbone_feature_dim: int = 512

    # Параметры проектора
    projection_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    projection_dropout: float = 0.3
    cnn_embedding_dim: int = 64

    # Параметры Fuzzy слоя
    num_rules: int = 16
    fuzzy_init_sigma: float = 1.0
    fuzzy_use_tsk1: bool = False

    # Параметры регуляризации
    backbone_dropout: float = 0.3
    dropout_rate: float = 0.3
    use_batch_norm: bool = True

    # Выходной слой
    num_classes: int = 1

    # Дополнительные параметры
    extra_params: Dict[str, Any] = field(default_factory=dict)


class CNNWithFuzzy(nn.Module):
    """
    Архитектура 1D CNN + Fuzzy
    """

    def __init__(self, config: FuzzyCNNConfig):
        super(CNNWithFuzzy, self).__init__()
        self.config = config

        # 1. Создаем 1D CNN бэкбон
        # Извлекаем параметры для DenseNet из extra_params
        extra_params = config.extra_params.copy() if config.extra_params else {}

        self.backbone = ModelFactory.create_model(
            model_name=config.backbone_name,
            input_channels=config.input_channels,
            input_width=config.input_width,
            feature_dim=config.backbone_feature_dim,
            dropout_rate=config.backbone_dropout,
            use_bn=config.use_batch_norm,
            **extra_params
        )

        self.backbone_feature_dim = config.backbone_feature_dim

        # 2. Cжатие признаков
        self.projection = self._build_projection()

        # 3. Создаем Fuzzy слой
        self.fuzzy_layer = FuzzyTSKLayer(
            num_features=config.cnn_embedding_dim,
            num_rules=config.num_rules,
            num_outputs=config.num_classes,
            init_sigma=config.fuzzy_init_sigma,
            use_tsk1=config.fuzzy_use_tsk1,
            dropout=config.dropout_rate
        )

        # 4. Выходная активация
        self.output_activation = nn.Sigmoid()

        # 5. Инициализация
        self._initialize_weights()

        # Общую информацию
        total_params = sum(p.numel() for p in self.parameters())

    def _build_projection(self):
        """Строит проектор для сжатия признаков"""
        layers = []
        current_dim = self.backbone_feature_dim

        for hidden_dim in self.config.projection_hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.config.projection_dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, self.config.cnn_embedding_dim))
        if self.config.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.config.cnn_embedding_dim))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Инициализация весов проектора"""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, 23, 2560) - 1D сигнал
        if x.dim() == 4:
            x = x.squeeze(2)

        # Извлечение признаков 1D CNN
        features = self.backbone(x)  # (batch, backbone_feature_dim)

        # Проекция
        embedding = self.projection(features)  # (batch, cnn_embedding_dim)

        # Нечеткий вывод
        fuzzy_output, rule_activations = self.fuzzy_layer(embedding)

        # Сигмоид для вероятности
        output = self.output_activation(fuzzy_output)

        return output, rule_activations, embedding

    def get_backbone_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.backbone.parameters())
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        info = {
            'name': self.config.backbone_name,
            'feature_dim': self.backbone_feature_dim,
            'input_channels': self.config.input_channels,
            'input_width': self.config.input_width,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout': self.config.backbone_dropout,
            'use_bn': self.config.use_batch_norm,
        }

        # Специфичные параметры
        if self.config.backbone_name == 'densenet':
            info['growth_rate'] = self.config.extra_params.get('growth_rate', 32)
            info['n_layers'] = self.config.extra_params.get('n_layers', 4)

        return info

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)