import torch
from vocab import Vocabulary
import evaluation_models

# for coco
print('Evaluation on COCO:')
evaluation_models.evalrank("pretrain_model/coco/model_coco_1.pth.tar", "pretrain_model/coco/model_coco_2.pth.tar", data_path='data/', split="testall", fold5=True)

# for flickr
print('Evaluation on Flickr30K:')
evaluation_models.evalrank("pretrain_model/flickr/model_fliker_1.pth.tar", "pretrain_model/flickr/model_fliker_2.pth.tar", data_path='data/', split="test", fold5=False)
