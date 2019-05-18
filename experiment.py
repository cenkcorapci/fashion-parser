import argparse
from experiments.unet_experiment import UNetExperiment

usage_docs = """
--epochs <integer> Number of epochs
--val_split <float> Set validation split(between 0 and 1)
--early_stopping_patience <Integer> Stop after nb of epochs without improvement on val score
--batch_size <integer> Batch size for the optimizer
--sample_size <integer> How many segmentation samples will it use
--image_size <integer> resize image to size x size
"""

parser = argparse.ArgumentParser(usage=usage_docs)

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--val_split', type=float, default=.1)
parser.add_argument('--early_stopping_patience', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--sample_size', type=int, default=None)
parser.add_argument('--image_size', type=int, default=512)

args = parser.parse_args()

model = UNetExperiment(batch_size=args.batch_size,
                       nb_epochs=args.epochs,
                       early_stopping_at=args.early_stopping_patience,
                       val_split=args.val_split,
                       width=args.image_size,
                       height=args.image_size,
                       debug_sample_size=args.sample_size)

model.train_model()
model.generate_submission()
