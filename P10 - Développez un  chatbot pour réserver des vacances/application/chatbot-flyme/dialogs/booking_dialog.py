# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Flight booking dialog."""

from datatypes_date_time.timex import Timex

from botbuilder.dialogs import WaterfallDialog, WaterfallStepContext, DialogTurnResult
from botbuilder.dialogs.prompts import ConfirmPrompt, TextPrompt, PromptOptions
from botbuilder.core import MessageFactory, BotTelemetryClient, NullTelemetryClient
from .cancel_and_help_dialog import CancelAndHelpDialog
from .start_date_resolver_dialog import StartDateResolverDialog
from .end_date_resolver_dialog import EndDateResolverDialog



class BookingDialog(CancelAndHelpDialog):
    """Flight booking implementation."""

    def __init__(
        self,
        dialog_id: str = None,
        telemetry_client: BotTelemetryClient = NullTelemetryClient(),
    ):
        super(BookingDialog, self).__init__(
            dialog_id or BookingDialog.__name__, telemetry_client
        )
        self.telemetry_client = telemetry_client
        text_prompt = TextPrompt(TextPrompt.__name__)
        text_prompt.telemetry_client = telemetry_client

        waterfall_dialog = WaterfallDialog(
            WaterfallDialog.__name__,
            [
                self.origin_step,
                self.destination_step,
                self.start_travel_date_step,
                self.end_travel_date_step,
                self.budget_step,
                self.confirm_step,
                self.final_step,
            ],
        )
        waterfall_dialog.telemetry_client = telemetry_client

        self.add_dialog(text_prompt)
        self.add_dialog(ConfirmPrompt(ConfirmPrompt.__name__))
        self.add_dialog(
            StartDateResolverDialog(StartDateResolverDialog.__name__, self.telemetry_client)
        )
        self.add_dialog(
            EndDateResolverDialog(EndDateResolverDialog.__name__, self.telemetry_client)
        )
        self.add_dialog(waterfall_dialog)

        self.initial_dialog_id = WaterfallDialog.__name__

        #Python dictionary used to save chat history
        self.chat_history = dict()

    async def origin_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for origin city."""
        self.chat_history["chat_request"] = step_context._turn_context.activity.text

        booking_details = step_context.options

        if booking_details.origin is None:
            return await step_context.prompt(
                TextPrompt.__name__,
                PromptOptions(
                    prompt=MessageFactory.text("From which city would you like to leave?")
                ),
            )  # pylint: disable=line-too-long,bad-continuation

        return await step_context.next(booking_details.origin)

    async def destination_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for destination city."""
        self.chat_history["chat_origin"] = step_context._turn_context.activity.text

        booking_details = step_context.options

        # Capture the response to the previous step's prompt
        booking_details.origin = step_context.result

        if booking_details.destination is None:
            return await step_context.prompt(
                TextPrompt.__name__,
                PromptOptions(
                    prompt=MessageFactory.text("To which city would you like to go?")
                ),
            )  # pylint: disable=line-too-long,bad-continuation

        return await step_context.next(booking_details.destination)


    async def start_travel_date_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for travel start date.
        This will use the START_DATE_RESOLVER_DIALOG."""
        self.chat_history["chat_destination"] = step_context._turn_context.activity.text

        booking_details = step_context.options

        # Capture the results of the previous step
        booking_details.destination = step_context.result

        if not booking_details.start_travel_date or self.is_ambiguous(
            booking_details.start_travel_date
        ):
            return await step_context.begin_dialog(
                StartDateResolverDialog.__name__, booking_details.start_travel_date
            )  # pylint: disable=line-too-long

        return await step_context.next(booking_details.start_travel_date)

    async def end_travel_date_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for travel end date.
        This will use the END_DATE_RESOLVER_DIALOG."""


        self.chat_history["chat_start_travel_date"] = step_context._turn_context.activity.text

        booking_details = step_context.options

        # Capture the results of the previous step
        booking_details.start_travel_date = step_context.result

        if not booking_details.end_travel_date or self.is_ambiguous(
            booking_details.end_travel_date
        ):
            return await step_context.begin_dialog(
                EndDateResolverDialog.__name__, booking_details.end_travel_date
            )  # pylint: disable=line-too-long

        return await step_context.next(booking_details.end_travel_date)

    async def budget_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Prompt for budget."""
        self.chat_history["chat_end_travel_date"] = step_context._turn_context.activity.text

        booking_details = step_context.options

        # Capture the response to the previous step's prompt
        booking_details.end_travel_date = step_context.result

        if booking_details.budget is None:
            return await step_context.prompt(
                TextPrompt.__name__,
                PromptOptions(
                    prompt=MessageFactory.text("What is your budget?")
                ),
            )  # pylint: disable=line-too-long,bad-continuation

        return await step_context.next(booking_details.budget)
        

    async def confirm_step(
        self, step_context: WaterfallStepContext
    ) -> DialogTurnResult:
        """Confirm the information the user has provided."""
        self.chat_history["chat_budget"] = step_context._turn_context.activity.text

        booking_details = step_context.options

        # Capture the results of the previous step
        booking_details.budget = step_context.result
        
        msg = (
            f"Please confirm, I have you traveling from: { booking_details.origin } to: { booking_details.destination }"
            f" from the: { booking_details.start_travel_date } to the: { booking_details.end_travel_date}"
            f" for a budget of: { booking_details.budget }."
       )

        # Offer a YES/NO prompt.
        return await step_context.prompt(
            ConfirmPrompt.__name__, PromptOptions(prompt=MessageFactory.text(msg))
        )

    async def final_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        """Complete and tracing for monitoring"""

        self.chat_history["chat_validation_by_user"] = step_context._turn_context.activity.text
        booking_details = step_context.options

        booking_info = {
            "origin": booking_details.origin,
            "destination": booking_details.destination,
            "start_travel_date": booking_details.start_travel_date,
            "end_travel_date": booking_details.end_travel_date,
            "budget": booking_details.budget,
        }

        if step_context.result:
            self.telemetry_client.track_metric("TransactionConfirmed", 1, properties=booking_info)
            self.telemetry_client.track_metric("ChatHistory", 1, properties=self.chat_history)
            self.telemetry_client.track_event("BookingConfirmed", properties=booking_details.__dict__)
        else:
            self.telemetry_client.track_metric("TransactionDismissed", 1, properties=booking_info, severity_level="ERROR")
            self.telemetry_client.track_metric("ChatError", 1, properties=self.chat_history, severity_level="ERROR")
            self.telemetry_client.track_event("BookingRefused", properties=booking_details.__dict__)

        self.telemetry_client.flush()
        return await step_context.end_dialog() 



    def is_ambiguous(self, timex: str) -> bool:
        """Ensure time is correct."""
        timex_property = Timex(timex)
        return "definite" not in timex_property.types
