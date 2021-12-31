# Dynamic-Ablation
This repo contains the supported code to reproduce attribution results of [Dynamic Ablation](https://github.com/ChopsticksAN/Dynamic-Ablation).
# Usage
```
python dynamic_ablation.py -a resnet50
```
- "resnet50" can be replaced with following networks from pretrainedmodels
```
print(pretrainedmodels.model_names)
> ['fbresnet152', 'bninception', 'resnext101_32x4d', 'resnext101_64x4d', 'inceptionv4', 'inceptionresnetv2', 'alexnet', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'inceptionv3', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'nasnetalarge', 'nasnetamobile', 'cafferesnet101', 'senet154',  'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'cafferesnet101', 'polynet', 'pnasnet5large']
```
