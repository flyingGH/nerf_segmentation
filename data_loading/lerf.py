import json
import numpy as np

def visualize_lerf_trajector(dir):

    with open(dir / 'transforms.json') as f:
        j = json.load(f)

    import pyviz3d.visualizer as viz
    v = viz.Visualizer()

    for i, frame in enumerate(j['frames'][::1]):
        c2w = np.array(frame['transform_matrix']).reshape(4,4)[0:3, :]
        origin = c2w @ np.array([0, 0, 0, 1])
        v.add_arrow(f'{i};Arrow_1', start=origin, end=c2w @ np.array([0.1, 0.0, 0.0, 1]), color=np.array([255, 0, 0]), stroke_width=0.005, head_width=0.01)
        v.add_arrow(f'{i};Arrow_2', start=origin, end=c2w @ np.array([0.0, 0.1, 0.0, 1]), color=np.array([0, 255, 0]), stroke_width=0.005, head_width=0.01)
        v.add_arrow(f'{i};Arrow_3', start=origin, end=c2w @ np.array([0.0, 0.0, 0.1, 1]), color=np.array([0, 0, 255]), stroke_width=0.005, head_width=0.01)
    v.save(dir / 'visualization')