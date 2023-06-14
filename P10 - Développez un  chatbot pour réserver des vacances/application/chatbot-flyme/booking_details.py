# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class BookingDetails:
    def __init__(
        self,
        destination: str = None,
        origin: str = None,
        travel_date: str = None,
        unsupported_airports=None,
    ):
        if unsupported_airports is None:
            unsupported_airports = []
        self.destination = destination
        self.origin = origin
        self.start_travel_date = travel_date
        self.end_travel_date = travel_date
        self.budget = travel_date
        self.travel_date = travel_date
        self.unsupported_airports = unsupported_airports



