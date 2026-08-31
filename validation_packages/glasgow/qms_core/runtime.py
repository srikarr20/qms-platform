from typing import Callable, Optional

from .state import (
    Measurement,
    TwinState,
)


class TwinRuntime:
    """
    Streaming detector-driven twin runtime.

    Each incoming Measurement is passed through a user-supplied
    reconstruction/update function:

        Measurement
            ->
        detector field
            ->
        virtual upstream state
            ->
        source estimate
            ->
        TwinState
    """

    def __init__(
        self,
        update_fn: Callable[[Measurement, Optional[TwinState]], TwinState],
    ):
        self.update_fn = update_fn

        self.state: Optional[TwinState] = None

        self.measurement_count = 0


    def update(
        self,
        measurement: Measurement,
    ) -> TwinState:

        new_state = self.update_fn(
            measurement,
            self.state,
        )

        self.measurement_count += 1

        new_state.version = self.measurement_count

        self.state = new_state

        return self.state


    def reset(self):

        self.state = None
        self.measurement_count = 0


    @property
    def current_state(self):

        return self.state
