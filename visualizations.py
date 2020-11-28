import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils import to_numpy
from dataset import mean, std

def plot(x_batch, out_batch, gt_batch, title="", fig=None):
    batch_size = x_batch.shape[0]
    num_classes = out_batch.shape[1]
    x_batch = to_numpy(x_batch)
    out_batch = to_numpy(out_batch)
    gt_batch = to_numpy(gt_batch)

    # cancal normalization
    # TODO: don't hardcode the numbers here
    x_batch = x_batch.transpose([0, 2, 3, 1])
    mean_np = np.array([[[[*mean]]]])
    std_np = np.array([[[[*std]]]])
    x_batch *= std_np
    x_batch += mean_np
    x_batch *= 255

    out_batch = out_batch.argmax(axis=1)

    if fig is None:
        fig = make_subplots(batch_size, 4)
    for i, x, y, gt in zip(range(1, batch_size+1), x_batch, out_batch, gt_batch):
        fig.add_trace(go.Image(z=x), i, 1)
        fig.add_trace(go.Heatmap(z=np.flipud(gt), zmin=0, zmax=num_classes - 1, showscale=False), i, 2)
        fig.add_trace(go.Heatmap(z=np.flipud(y), zmin=0, zmax=num_classes - 1, showscale=False), i, 3)
        fig.add_trace(go.Heatmap(z=np.flipud(y == gt).astype(np.float), showscale=False), i, 4)
        pass
    fig.update_layout(title_text=title)
    fig.show()
    return fig