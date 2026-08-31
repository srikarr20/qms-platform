from pathlib import Path
import json
import shutil
import sys

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parent.parent
INBOX = ROOT / "qms_pyxel_twin_009" / "inbox"
ARCHIVE = ROOT / "qms_pyxel_twin_009" / "archive"


def validate_measurement(folder):
    pixel = folder / "detector_pixel.npy"
    image = folder / "detector_image.fits"

    if not pixel.exists():
        raise FileNotFoundError(pixel)

    if not image.exists():
        raise FileNotFoundError(image)

    pixel_arr = np.load(pixel)
    image_arr = fits.getdata(image)

    if pixel_arr.shape != image_arr.shape:
        raise ValueError(
            f"Shape mismatch: pixel={pixel_arr.shape}, image={image_arr.shape}"
        )

    return pixel_arr, image_arr


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python ingest_measurement.py <measurement-directory>"
    )

source = Path(sys.argv[1]).resolve()

pixel, image = validate_measurement(source)

measurement_id = source.name

target = INBOX / measurement_id

if target.exists():
    shutil.rmtree(target)

shutil.copytree(source, target)

metadata = {
    "measurement_id": measurement_id,
    "pixel_shape": list(pixel.shape),
    "image_shape": list(image.shape),
    "pixel_dtype": str(pixel.dtype),
    "image_dtype": str(image.dtype),
    "source": str(source),
}

(target / "measurement.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)

print()
print("=== QMS EXTERNAL MEASUREMENT INGEST ===")
print()
print("measurement:", measurement_id)
print("pixel shape:", pixel.shape)
print("image shape:", image.shape)
print("accepted:", target)
