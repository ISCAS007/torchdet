## overlap
```
python model/overlap.py --save_model --batch_size 2
```
## model/overlap.py
the main funcion

- class seg_metric
input: tensor or numpy, BxHxW, label_True.size=label_pred.size
output: dict

- class trainer
input: config, edict
output: -

- load dataset
config.root_path --> coco/PennFudanDataset --> Det2Seg --> DataLoader --> trainer --> train_val --> train


