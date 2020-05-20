import tensorflow as tf
from matplotlib import pyplot as plt
from .plots import *
from tensorflow.keras import backend as K

class TrainingHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, metrics, class_names, save_weights=True, metrics_plot_dir=None):
        self.metrics = {}
        self.weights = []
        self.class_names = class_names
        self.save_weights = save_weights
        self.metrics_plot_dir = metrics_plot_dir
        for m in metrics:
            self.metrics[m] = []
            self.metrics['val_' + m] = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key in self.metrics.keys():
                if key == 'lr':
                    self.metrics[key].append(K.get_value(self.model.optimizer.lr))
                else:
                    self.metrics[key].append(logs.get(key))

        if self.metrics_plot_dir:
            for key in self.metrics.keys():
                if not key.startswith('val_'):
                    if key == 'f1_score':
                        plot_f1(self.metrics, self.class_names, self.metrics_plot_dir)
                        plot_f1_html(self.metrics, self.class_names, self.metrics_plot_dir)
                    else:
                        plot_metric(self.metrics, key, self.metrics_plot_dir)
                        plot_metric_html(self.metrics, key, self.metrics_plot_dir)

        if self.save_weights:
            self.weights.append([])
            for l in self.model.layers:
                self.weights[len(self.weights)-1].append(l.get_weights())