import io
import os
import shutil

import numpy as np
import tensorflow as tf
from PIL import Image

from commons.config import *


class TensorBoardMonitoring:
    def __init__(self, logdir):
        try:
            for the_file in os.listdir(logdir):
                file_path = os.path.join(logdir, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logging.error(e)
        except Exception as exp:
            logging.error(exp)
        finally:
            pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
        self._writer = tf.summary.FileWriter(logdir)

    def close(self):
        self._writer.close()

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self._writer.add_summary(summary, global_step=global_step)
        self._writer.flush()

    def log_histogram(self, tag, values, global_step, bins):
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary()
        summary.value.add(tag=tag, histo=hist)
        self._writer.add_summary(summary, global_step=global_step)
        self._writer.flush()

    def log_image(self, tag, img, global_step):
        s = io.BytesIO()
        Image.fromarray(img).save(s, format='png')

        img_summary = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self._writer.add_summary(summary, global_step=global_step)
        self._writer.flush()

    def log_plot(self, tag, figure, global_step):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                       height=img_ar.shape[0],
                                       width=img_ar.shape[1])

        summary = tf.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self._writer.add_summary(summary, global_step=global_step)
        self._writer.flush()
