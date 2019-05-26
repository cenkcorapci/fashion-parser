import argparse

from experiments.mask_rcnn_experiment import MaskRCNNExperiment

usage_docs = """
--epochs <integer> Number of epochs
--val_split <float> Set validation split(between 0 and 1)
--batch_size <integer> Batch size for the optimizer
--sample_size <integer> How many segmentation samples will it use
"""

parser = argparse.ArgumentParser(usage=usage_docs)

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--val_split', type=float, default=.1)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--sample_size', type=int, default=None)

args = parser.parse_args()

model = MaskRCNNExperiment(batch_size=args.batch_size,
                           nb_epochs=args.epochs,
                           val_split=args.val_split,
                           debug_sample_size=args.sample_size)

model.train_model()
model.predict()
