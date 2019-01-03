# Train a Segmentation Model for Coastline Detection

This implements training, recovering, and evaluating a Unet model traning process.

## Requirements  
- Install PyTorch ([pytorch.org](http://pytorch.org))
- Install python libraries
- `pip install -r requirements.txt`
 
## Data 
This implementation accepts a pandas pickle table with data information for train, validation, and evaluation.

The dataset should include columns with the names: "path", "name", "label", and "split".  
Format:

```bash
path: string
The path of an input image.

name: string
The image name of a train/validation/test image.

label: string
The path of a target mask image. [None] for evaluation split.

split: string
choice('train', 'val', 'test')
```



#### Example

![alt text][example_pic]

[example_pic]: https://github.com/tank145161/coastline/blob/master/dataExample.png "dataset example"

## Run
To train a model, `run main.py` with a DataFrame containing the 'path', 'label' and 'split' columns of each samples.

```bash
python main.py "./data/coastline.p" --arch=unet 
```
To recover training at specific Epoch, `run main.py` with a model path with '--resume' parameter. Can also append log information to past with '--resume-log'.

```bash
python main.py "./data/coastline.p" --arch=unet --resume=/path/to/model.pth.tar
```
To evaluate samples using a pre-trained model, raise '--evaluate' flag.

```bash
python main.py "./data/coastline.p" --arch=unet --resume=/path/to/model.pth.tar --evaluate
```

## Usage

```
usage: main.py [-h] [-a ARCH] [--input-dim N] [--output-dim N] [-s PATH] [-m PATH] [--resume PATH] [-e] [-p] [-j N] [--epochs N]
               [--start-epoch N] [-b N] [--lr LR] [--lr-deGamma LR] [--lr-deStep LR] [--momentum M] [--wd W] [--gpu GPU]
               DIR

PyTorch-Unet Binary Classification Training

positional arguments:
  DIR                   dataframe path to dataset include: index path label split

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture
  --input-dim N         number of channels of input data (default: 3)
  --output-dim N        number of channels of output data (class numbers) (default: 1)
  -s PATH, --save-folder PATH
                        folder path to save model files and log files (default: ./model)
  -m PATH, --save-prefix PATH
                        prefix string to specify version or task type (default: [date])
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  -p, --pretrained      use pre-trained model
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run (default: 100)
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel
                        or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.01)
  --lr-deGamma LR       learning rate deGamma (default: 0.1)
  --lr-deStep LR        initial learning rate deStep (default: 50)
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --gpu GPU             GPU id(s) to use.Format: a comma-delimited list
```

