# fashion-parser
[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/cenkcorapci/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/cenkcorapci/fashion-parser.svg)

Fashion segmentation based on [Matterport's mask r-cnn implementation](https://github.com/matterport/Mask_RCNN) using
[imaterialist-fashion-2019-FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/overview) data set.

## Usage
- Edit *commons/config.py* for workspace details, folder paths etc.
- Hyper parameters are set in *commons/fashion_config.py* which is inherited from [Matterport's config](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py)
- Use *experiment.py* to train and create a submission file;
```bash
python experiment.py --epochs <number of epochs> --val_split <between 0 - 1, ratio of validation samples>
```
## Example Results
![Alt text](examples/example_001.jpeg?raw=true "Title")
![Alt text](examples/example_002.jpeg?raw=true "Title")

## TODO
- [X] Switch to Keras
- [X] Change to Mask R-CNN
- [X] Update Readme
- [ ] Serve with Flask
