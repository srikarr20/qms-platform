import io
import zipfile
from pathlib import Path

import numpy as np


class GlasgowCumulativeArchive:
    """
    Reader for the Glasgow Heralded Diffraction SM dataset.

    Source files are cumulative 512x512 detector-count states:

        F_1, F_2, ..., F_N

    Real detector increments are recovered as:

        D_t = F_t - F_{t-1}

    We preserve only non-negative increments. A negative
    increment indicates that the cumulative assumption has
    been violated and raises an error.
    """

    def __init__(self, zip_path):
        self.zip_path = Path(zip_path).expanduser()

        if not self.zip_path.exists():
            raise FileNotFoundError(
                f"Glasgow archive not found: {self.zip_path}"
            )


    @staticmethod
    def _frame_number(name):
        stem = Path(name).stem

        try:
            return int(stem)
        except ValueError:
            return None


    def frame_names(self):
        with zipfile.ZipFile(
            self.zip_path,
            "r",
        ) as z:

            names = []

            for name in z.namelist():
                if not name.lower().endswith(".asc"):
                    continue

                number = self._frame_number(name)

                if number is not None:
                    names.append(
                        (
                            number,
                            name,
                        )
                    )

        names.sort(
            key=lambda x: x[0]
        )

        return [
            name
            for _, name in names
        ]


    @staticmethod
    def _decode_frame(raw):
        text = raw.decode(
            "ascii",
            errors="ignore",
        )

        frame = np.loadtxt(
            io.StringIO(text)
        )

        if frame.ndim != 2:
            raise ValueError(
                f"Expected 2D detector frame, got {frame.shape}"
            )

        return frame.astype(
            np.float64,
            copy=False,
        )


    def iter_cumulative_frames(self):
        names = self.frame_names()

        with zipfile.ZipFile(
            self.zip_path,
            "r",
        ) as z:

            for index, name in enumerate(names):

                frame = self._decode_frame(
                    z.read(name)
                )

                info = z.getinfo(name)

                yield {
                    "index": index,
                    "name": name,
                    "frame": frame,
                    "timestamp": info.date_time,
                }


    def iter_increments(self):
        """
        Yield one real acquisition-step increment at a time.

        The first cumulative zero frame is used as the baseline
        and does not itself produce an increment.
        """

        previous = None
        previous_record = None

        for record in self.iter_cumulative_frames():

            current = record["frame"]

            if previous is None:
                previous = current
                previous_record = record
                continue

            delta = current - previous

            minimum = float(
                np.min(delta)
            )

            if minimum < 0:
                raise ValueError(
                    "Negative detector increment found between "
                    f"{previous_record['name']} and {record['name']}: "
                    f"min={minimum}"
                )

            yield {
                "index": record["index"] - 1,

                "from_name":
                    previous_record["name"],

                "to_name":
                    record["name"],

                "increment":
                    delta,

                "added_count":
                    float(
                        np.sum(delta)
                    ),

                "changed_pixels":
                    int(
                        np.count_nonzero(delta)
                    ),

                "timestamp":
                    record["timestamp"],
            }

            previous = current
            previous_record = record
