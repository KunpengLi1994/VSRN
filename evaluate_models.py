import torch
from vocab import Vocabulary
import evaluation_models


evaluation_models.evalrank("pretrain_model/coco/model_coco_1.pth.tar", "pretrain_model/coco/model_coco_2.pth.tar", data_path='../SCAN_my/data/', split="testall", fold5=True)
