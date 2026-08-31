from pathlib import Path
import json
import shutil

import numpy as np


def latest_file(
    root,
    filename,
) -> Path:

    root = Path(root)

    files = sorted(
        root.rglob(
            filename
        )
    )

    if not files:
        raise FileNotFoundError(
            f"{filename} not found under {root}"
        )

    return files[-1]


def load_pixel(
    root,
):
    """
    Load detector Pixel representation.
    """
    path = latest_file(
        root,
        "detector_pixel.npy",
    )

    return np.asarray(
        np.load(path),
        dtype=np.float64,
    )


def load_image(
    root,
):
    """
    Load detector Image representation.
    """
    path = latest_file(
        root,
        "detector_image.fits",
    )

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError(
            "Loading FITS detector images requires "
            "the optional 'astropy' dependency."
        ) from exc

    return np.asarray(
        fits.getdata(path),
        dtype=np.float64,
    )


def validate_measurement_directory(
    folder,
):
    """
    Validate the QMS detector-measurement contract:

        detector_pixel.npy
        detector_image.fits
    """
    folder = Path(folder)

    pixel_path = (
        folder
        / "detector_pixel.npy"
    )

    image_path = (
        folder
        / "detector_image.fits"
    )

    if not pixel_path.exists():
        raise FileNotFoundError(
            pixel_path
        )

    if not image_path.exists():
        raise FileNotFoundError(
            image_path
        )

    pixel = np.load(
        pixel_path
    )

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError(
            "Validating FITS detector measurements requires "
            "the optional 'astropy' dependency."
        ) from exc

    image = fits.getdata(
        image_path
    )

    if pixel.shape != image.shape:
        raise ValueError(
            "Shape mismatch: "
            f"pixel={pixel.shape}, "
            f"image={image.shape}"
        )

    return pixel, image


def ingest_measurement(
    source,
    inbox,
    *,
    replace: bool = True,
):
    """
    Copy a detector measurement into a QMS inbox
    and write machine-readable metadata.
    """
    source = Path(
        source
    ).resolve()

    inbox = Path(
        inbox
    )

    inbox.mkdir(
        parents=True,
        exist_ok=True,
    )

    pixel, image = (
        validate_measurement_directory(
            source
        )
    )

    measurement_id = (
        source.name
    )

    target = (
        inbox
        / measurement_id
    )

    if target.exists():

        if not replace:
            raise FileExistsError(
                target
            )

        shutil.rmtree(
            target
        )

    shutil.copytree(
        source,
        target,
    )

    metadata = {
        "measurement_id":
            measurement_id,

        "pixel_shape":
            list(
                pixel.shape
            ),

        "image_shape":
            list(
                image.shape
            ),

        "pixel_dtype":
            str(
                pixel.dtype
            ),

        "image_dtype":
            str(
                image.dtype
            ),

        "source":
            str(
                source
            ),
    }

    (
        target
        / "measurement.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n"
    )

    return target, metadata


def archive_measurement(
    folder,
    archive,
    *,
    replace: bool = True,
):
    folder = Path(
        folder
    )

    archive = Path(
        archive
    )

    archive.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = (
        archive
        / folder.name
    )

    if target.exists():

        if not replace:
            raise FileExistsError(
                target
            )

        shutil.rmtree(
            target
        )

    shutil.move(
        str(folder),
        str(target),
    )

    return target


def load_twin_state(
    path,
):
    path = Path(
        path
    )

    if path.exists():
        return json.loads(
            path.read_text()
        )

    return {
        "history": []
    }


def save_twin_state(
    path,
    state,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            state,
            indent=2,
        )
        + "\n"
    )
