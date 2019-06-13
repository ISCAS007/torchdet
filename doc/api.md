## overlap
```
python model/overlap.py --save_model --batch_size 2
```
## model/overlap.py
the main funcion

- class seg_metric
input: tensor or numpy, BxHxW, label_True.size=label_pred.size
output: dict

## segmentation dataset
- torchvision.datasets.VOCSegmentation
output PIL.Image, need convert to numpy array
- dataset information flow
```
digraph G{
{config split} -> dataset
split -> transform
{dataset transform} -> seg2boundary
seg2boundary -> dataloader
}
```