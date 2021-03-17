# Yolov4 for Detection

### Related Repository

https://github.com/argusswift/YOLOv4-pytorch

### Envrioments and Datasets

`pip install opencv-python`

`pip install tensorboardX`

`pip install pycocotools`

`cd data`

`ln -s /path/to/VOCdevkit`

`cd ..`

`vi config/yolov4_config.py` Then modify the DETECTION_PATH

`cd utils`

`python voc.py`

`cd ..`

`cd weight` Then download pre-trained weights from [here](https://www.dropbox.com/sh/uwois7q7b6mfdg4/AAD493jEVwHB9A8RQPFiOeu0a?dl=0).

`cd ..` 

Then put your pruned masks somewhere and modify the `MASKROOT`  in the cmd scripts. 

### Iterative Magnitude Pruning (IMP)

`bash TicketSH/imp_imagenet.sh 0`

`bash TicketSH/imp_simclr.sh 0`

`bash TicketSH/imp_moco.sh 0`

### Transfer Experiments

`bash TicketSH/transfer.sh 0 1`

Remark. The first integer number 0 indicates the index of used GPU, and the second one denotes the sparsity level of the corresponding mask.

### Random Pruning

`nohup bash random/01_random_moco0102.sh 0 > Random0102.log &`

Remark. The integer number 0 indicates the index of used GPU.

 

