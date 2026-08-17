import numpy as np

from .reconstruction import propagate


def spatial_metrics(field, X, Y):
    I = np.abs(field) ** 2
    total = I.sum() + 1e-15

    cx = float(np.sum(I * X) / total)
    cy = float(np.sum(I * Y) / total)

    width = float(
        np.sqrt(
            np.sum(
                I * (
                    (X - cx) ** 2
                    + (Y - cy) ** 2
                )
            ) / total
        )
    )

    return cx, cy, width


def blind_depth_search(
    sensor_field,
    depths,
    FX,
    FY,
    wavelength,
    X,
    Y,
):
    widths = np.zeros(len(depths))
    centroids = np.zeros((len(depths), 2))
    volume = []

    for i, z in enumerate(depths):
        field = propagate(
            sensor_field,
            FX,
            FY,
            wavelength,
            -z,
        )

        volume.append(field)

        cx, cy, width = spatial_metrics(
            field,
            X,
            Y,
        )

        centroids[i] = [cx, cy]
        widths[i] = width

    best = int(np.argmin(widths))

    return {
        "best_index": best,
        "best_depth": float(depths[best]),
        "x": float(centroids[best, 0]),
        "y": float(centroids[best, 1]),
        "widths": widths,
        "centroids": centroids,
        "volume": np.asarray(volume),
    }
