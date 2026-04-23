"""
VGG models for 1D signal classification.
"""

import torch
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, features, ngpu, num_classes=4, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.ngpu = ngpu
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.ReLU(True),        
            nn.Dropout(),        
            nn.Linear(256, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            x = nn.parallel.data_parallel(self.features, x, range(self.ngpu))
            x = x.view(x.size(0), -1)
            x = nn.parallel.data_parallel(self.classifier, x, range(self.ngpu))
        else:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d): 
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 1 
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=3, stride=3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'E': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
}


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


# =============================================================================
# Wrapper class for compatibility with model_adapt.py (ResNet interface)
# =============================================================================

class VGGFeatureExtractor(nn.Module):
    """
    Wrapper around VGG to provide ResNet-compatible interface for model_adapt.py.
    
    This class wraps the VGG feature extraction layers only (no classifier),
    and provides the same interface as ResNet1D.
    """
    
    def __init__(self, signal_channels, stride=2, layer_norm=False, feat_dim=256, 
                 dropout_p=0.5, use_attention=False, attention_heads=8, 
                 attention_type="self", vgg_config='D', **kwargs):
        super(VGGFeatureExtractor, self).__init__()
        
        self.feat_dim = feat_dim
        self.layer_norm = layer_norm
        
        # Build VGG features with custom in_channels
        self.features = self._make_layers(cfg[vgg_config], in_channels=signal_channels, batch_norm=True)
        
        # Get output channels from last conv layer
        self.feature_channels = self._get_feature_channels()
        
        # Adaptive pooling and FC to match feat_dim
        self.avgpool = nn.AdaptiveAvgPool1d(2)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_channels * 2, feat_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
        )
        
        if self.layer_norm:
            self.feat_norm_layer = nn.LayerNorm(feat_dim)
        
        # For compatibility with model_adapt.py
        self.z_backbone = None
        self.z_att = None
        
        self._initialize_weights()
    
    def _make_layers(self, cfg_list, in_channels=1, batch_norm=True):
        layers = []
        for v in cfg_list:
            if v == 'M':
                layers += [nn.MaxPool1d(kernel_size=3, stride=3)]
            else:
                conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv1d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def _get_feature_channels(self):
        for module in reversed(list(self.features.modules())):
            if isinstance(module, nn.Conv1d):
                return module.out_channels
        return 256
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d): 
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                m.bias.data.zero_()
    
    def forward(self, x, sqi_sample=None):
        """
        Forward pass compatible with ResNet1D interface.
        
        Args:
            x: Input tensor of shape (B, C, T)
            sqi_sample: SQI sample tensor (ignored, for compatibility)
            
        Returns:
            Feature tensor of shape (B, feat_dim)
        """
        x = self.features(x)
        
        # Store backbone embedding
        self.z_backbone = torch.mean(x, dim=-1)  # (B, C)
        self.z_att = None
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if self.layer_norm:
            x = self.feat_norm_layer(x)
        
        return x


def vgg16_1D(signal_channels, stride=2, layer_norm=False, feat_dim=256, dropout_p=0.5,
             use_attention=False, attention_heads=8, attention_type="self", **kwargs):
    """
    VGG16 feature extractor with ResNet-compatible interface.
    Can be used as drop-in replacement for resnet18_1D/resnet34_1D in model_adapt.py.
    """
    return VGGFeatureExtractor(
        signal_channels=signal_channels,
        stride=stride,
        layer_norm=layer_norm,
        feat_dim=feat_dim,
        dropout_p=dropout_p,
        use_attention=use_attention,
        attention_heads=attention_heads,
        attention_type=attention_type,
        vgg_config='D',
        **kwargs
    )


def vgg19_1D(signal_channels, stride=2, layer_norm=False, feat_dim=256, dropout_p=0.5,
             use_attention=False, attention_heads=8, attention_type="self", **kwargs):
    """
    VGG19 feature extractor with ResNet-compatible interface.
    Can be used as drop-in replacement for resnet18_1D/resnet34_1D in model_adapt.py.
    """
    return VGGFeatureExtractor(
        signal_channels=signal_channels,
        stride=stride,
        layer_norm=layer_norm,
        feat_dim=feat_dim,
        dropout_p=dropout_p,
        use_attention=use_attention,
        attention_heads=attention_heads,
        attention_type=attention_type,
        vgg_config='E',
        **kwargs
    )
