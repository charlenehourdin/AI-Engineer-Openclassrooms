# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from enum import Enum
from typing import Dict
from botbuilder.ai.luis import LuisRecognizer
from botbuilder.core import IntentScore, TopIntent, TurnContext

from booking_details import BookingDetails


class Intent(Enum):
    BOOK_FLIGHT = "OrderTravelIntent"
    GREET_INTENT = "GreetingsIntent"
    NONE_INTENT = "None"


def top_intent(intents: Dict[Intent, dict]) -> TopIntent:
    max_intent = Intent.NONE_INTENT
    max_value = 0.0

    for intent, value in intents:
        intent_score = IntentScore(value)
        if intent_score.score > max_value:
            max_intent, max_value = intent, intent_score.score

    return TopIntent(max_intent, max_value)


class LuisHelper:
    @staticmethod
    async def execute_luis_query(
        luis_recognizer: LuisRecognizer, turn_context: TurnContext
    ) -> (Intent, object):
        """
        Returns an object with preformatted LUIS results for the bot's dialogs to consume.
        """
        result = None
        intent = None

        try:
            recognizer_result = await luis_recognizer.recognize(turn_context)

            intent = (
                sorted(
                    recognizer_result.intents,
                    key=recognizer_result.intents.get,
                    reverse=True,
                )[:1][0]
                if recognizer_result.intents
                else None
            )

            if intent == Intent.BOOK_FLIGHT.value:
                result = BookingDetails()

                # We need to get the result from the LUIS JSON which at every level returns an array.
                from_entities = recognizer_result.entities.get("$instance", {}).get("DepartureCity", [])
                if len(from_entities) > 0:
                    result.origin = from_entities[0]["text"].capitalize()

                #destination
                to_entities = recognizer_result.entities.get("$instance", {}).get("ArrivalCity", [])
                if len(to_entities) > 0:
                    result.destination = to_entities[0]["text"].capitalize()

                # This value will be a TIMEX. And we are only interested in a Date so grab the first result and drop
                # the Time part. TIMEX is a format that represents DateTime expressions that include some ambiguity.
                # e.g. missing a Year.
                date_entities = recognizer_result.entities.get("datetime", [])
                if date_entities:
                    if len(date_entities)==1:
                        timex = date_entities[0]["timex"]
                        if date_entities[0]['type'] == 'daterange':
                            datetime_range = timex[0].strip('(').strip(')').split(',')
                            result.start_travel_date = datetime_range[0]
                            result.end_travel_date = datetime_range[1]
                        elif date_entities[0]['type'] == 'date':
                            result.start_travel_date = timex[0]
                    
                    elif len(date_entities)==2:
                        timex1 = date_entities[0]["timex"]
                        timex2 = date_entities[1]["timex"]
                        if timex1[0] <= timex2[0]:
                            result.start_travel_date = timex1[0]
                            result.end_travel_date = timex2[0]
                        else:
                            result.start_travel_date = timex2[0]
                            result.end_travel_date = timex1[0]

                            
                #budget
                budget_entities = recognizer_result.entities.get("$instance", {}).get("Price", [])
                if len(budget_entities) > 0:
                    result.budget = budget_entities[0]["text"]

        except Exception as exception:
            print(exception)

        return intent, result
