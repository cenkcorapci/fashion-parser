# fashion-parser
[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/cenkcorapci/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/cenkcorapci/fashion-parser.svg)

Fashion segmentation based on [Matterport's mask r-cnn implementation](https://github.com/matterport/Mask_RCNN) using
[imaterialist-fashion-2019-FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/overview) data set.

## Requirements

This project has been updated to use modern Python libraries:
- Python 3.12+
- TensorFlow 2.13+
- For full dependencies, see `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage
- Edit *commons/config.py* for workspace details, folder paths etc.
- Hyper parameters are set in *commons/fashion_config.py* which is inherited from [Matterport's config](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py)
- Use *experiment.py* to train and create a submission file;
```bash
python experiment.py --epochs <number of epochs> --val_split <between 0 - 1, ratio of validation samples>
```

## Testing

The project now includes comprehensive unit tests. To run the tests:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_image_utils.py -v
```

Test coverage includes:
- Image utility functions (resize, RLE encoding, mask refinement)
- Data loading and preprocessing
- Configuration classes
- Dataset preparation logic

## Example Results
![Alt text](examples/example_001.jpeg?raw=true "Title")
![Alt text](examples/example_002.jpeg?raw=true "Title")

## Recent Updates
- ✅ Upgraded to TensorFlow 2.x and modern dependencies
- ✅ Updated Keras imports to use TensorFlow 2.x integrated Keras API
- ✅ Added comprehensive unit tests (38 tests)
- ✅ Fixed pandas API compatibility issues
- ✅ Improved configuration handling for test environments

## TODO
- [X] Switch to Keras
- [X] Change to Mask R-CNN
- [X] Update Readme
- [X] Add unit tests
- [ ] Serve with Flask
