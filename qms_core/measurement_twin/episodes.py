from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class AlertEpisode:
    episode: int
    start_window: int
    end_window: int
    duration_increments: int
    raw_flag_count: int
    peak_window: int
    peak_innovation: float
    peak_frozen_z: Optional[float]


def consolidate_alerts(
    alerts: Sequence,
    *,
    refractory: int,
):
    """
    Consolidate flagged windows into connected alert
    episodes using a maximum inter-flag gap.

    Important:
    These are connected alert clusters. They are NOT
    asserted to be statistically independent events.
    """
    flags = sorted(
        [
            alert
            for alert in alerts
            if alert.flagged
        ],
        key=lambda a:
            a.window_end,
    )

    if not flags:
        return []

    groups = []

    current = [
        flags[0]
    ]

    for alert in flags[1:]:

        gap = (
            alert.window_end
            - current[-1].window_end
        )

        if gap <= refractory:
            current.append(
                alert
            )

        else:
            groups.append(
                current
            )

            current = [
                alert
            ]

    groups.append(
        current
    )

    episodes = []

    for index, group in enumerate(
        groups,
        start=1,
    ):
        strongest = max(
            group,
            key=lambda a:
                a.innovation_norm,
        )

        start = int(
            group[0].window_end
        )

        end = int(
            group[-1].window_end
        )

        episodes.append(
            AlertEpisode(
                episode=index,

                start_window=start,

                end_window=end,

                duration_increments=(
                    end - start
                ),

                raw_flag_count=(
                    len(group)
                ),

                peak_window=int(
                    strongest.window_end
                ),

                peak_innovation=float(
                    strongest.innovation_norm
                ),

                peak_frozen_z=(
                    strongest.robust_z_frozen
                ),
            )
        )

    return episodes
