from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_metric(metrics, metric_name, save_dest=None):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(metrics[metric_name])
    if 'val_' + metric_name in metrics:
        plt.plot(metrics['val_' + metric_name])
        plt.legend(['Training', 'Validation'], loc='upper left')
    else:
        plt.legend(['Training'], loc='upper left')
    plt.title(metric_name)
    plt.ylabel(metric_name)
    plt.xlabel('epoch')

    if metric_name == 'loss':
        plt.ylim(top=10, bottom=0) 

    if save_dest:
        plt.savefig(os.path.join(save_dest, metric_name + '.png'))
    else:
        plt.show()
    plt.close(fig)


def plot_f1(metrics, class_names, save_dest=None):
    fig, m_axs = plt.subplots(2, 1, figsize=(12, 16))

    m_axs[0].plot(metrics['f1_score'])
    m_axs[0].set_ylabel('f1_score')
    m_axs[0].set_xlabel('epoch')
    m_axs[0].set_title('Training')
    m_axs[0].legend(class_names, loc='upper left')

    m_axs[1].plot(metrics['val_f1_score'])
    m_axs[1].set_ylabel('val_f1_score')
    m_axs[1].set_xlabel('epoch')
    m_axs[1].set_title('Validation')
    m_axs[1].legend(class_names, loc='upper left')

    if save_dest:
        fig.savefig(os.path.join(save_dest, 'f1_score.png'))
    else:
        plt.show()
    plt.close(fig)

    plot_metric({
        'f1_score_average': np.array(metrics['f1_score']).mean(axis=1),
        'val_f1_score_average': np.array(metrics['val_f1_score']).mean(axis=1)},
        'f1_score_average',
        save_dest)


def plot_metric_html(metrics, metric_name, save_dest):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=metrics[metric_name], mode='lines+markers', name=metric_name))
    val_metric_name = 'val_' + metric_name
    if val_metric_name in metrics:
        fig.add_trace(go.Scatter(y=metrics[val_metric_name], mode='lines+markers', name=val_metric_name))

    fig.update_layout(
        title=metric_name,
        xaxis_title='epoch',
        yaxis_title=metric_name
    )
    fig.write_html(os.path.join(save_dest, metric_name + '.html'))


def plot_f1_html(metrics, class_names, save_dest):
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Training", "Validation"))

    metric_name = 'f1_score'
    metric_arr = np.array(metrics[metric_name])
    class_number = metric_arr.shape[1]
    for i in range(0, class_number):
        fig.add_trace(
            go.Scatter(y=metric_arr[:,i], mode='lines+markers', name=class_names[i]),
            row=1, col=1
        )

    val_metric_name = 'val_' + metric_name
    val_metric_arr = np.array(metrics[val_metric_name])
    for i in range(0, class_number):
        fig.add_trace(
            go.Scatter(y=val_metric_arr[:,i], mode='lines+markers', name='val ' + class_names[i]),
            row=2, col=1
        )

    fig.update_xaxes(title_text='epoch', row=1, col=1)
    fig.update_yaxes(title_text='f1_score', row=1, col=1)
    fig.update_xaxes(title_text='epoch', row=2, col=1)
    fig.update_yaxes(title_text='val_f1_score', row=2, col=1)

    fig.write_html(os.path.join(save_dest, metric_name + '.html'))

    plot_metric_html({
        'f1_score_average': np.array(metrics['f1_score']).mean(axis=1),
        'val_f1_score_average': np.array(metrics['val_f1_score']).mean(axis=1)},
        'f1_score_average',
        save_dest)

def plot_confusion_matrix(true_lables, pred_labels, target_names, save_dest=None):
    cm = confusion_matrix(true_lables, pred_labels)
    cmap = plt.get_cmap('Blues')
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_dest:
        fig.savefig(os.path.join(save_dest, 'confusion_matrix.png'))
    else:
        plt.show()
    plt.close(fig)


def show_batch(image_batch, label_batch, label_names, number_to_show=4, predicted_labels=None):
    row_count = math.ceil(number_to_show / 4)
    fig, m_axs = plt.subplots(row_count, 4, figsize=(16, row_count * 4))
    for i, (c_x, c_y, c_ax) in enumerate(zip(image_batch, label_batch, m_axs.flatten())):
        c_ax.imshow(c_x)
        real_level = label_names[c_y == 1][0]
        pred_level = ''
        title = 'Real level: ' + real_level
        if predicted_labels is not None:
            pred_level = label_names[predicted_labels[i]]
            title = title + '\nPredicted one: ' + pred_level
        c_ax.set_title(title, color='g' if pred_level ==
                       '' or real_level == pred_level else 'r')
        c_ax.axis('off')

def create_confusion_matrix(model, dataset, steps, target_names, save_dest=None):
    print('Creating confusion matrix.')
    it = iter(dataset)
    true_labels_glob = []
    pred_labels_glob = []

    for i in range(0, steps):
        image_batch, true_labels = next(it)
        true_labels_glob.extend(np.argmax(true_labels, axis=1))
        pred = model.predict(image_batch)
        pred_labels_glob.extend(np.argmax(pred, axis=1))

    plot_confusion_matrix(
        true_labels_glob, pred_labels_glob, target_names, save_dest)
    print('Confusion matrix was saved to', save_dest)