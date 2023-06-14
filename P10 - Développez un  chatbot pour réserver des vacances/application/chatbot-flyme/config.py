#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Configuration for the bot."""

import os

class DefaultConfig:
    """Configuration for the bot."""

    #WEB APP CONFIGURATION
    PORT = 8000
    #APP_ID = None
    #APP_PASSWORD = None
    APP_ID =os.environ.get('APP_ID', '6ca3c0a8-cb4e-4c59-8398-5f86cabeb464')
    APP_PASSWORD = os.environ.get('APP_PASSWORD', 'Fbk8Q~I-4eNpz.c0sJbFaaZ5mHAKhiMFByk2LalM')
    
    #LUIS APP CONFIGURATION
    LUIS_APP_ID = os.environ.get("LUIS_APP_ID", 'd80f7484-6fa1-4229-b878-ef0f7f5c832c')
    LUIS_API_KEY = os.environ.get("LUIS_API_KEY", '2c2e5515ac8c4f3a803028b146f5d0d9')
    LUIS_API_HOST_NAME = os.environ.get("LUIS_API_HOST_NAME", 'westeurope.api.cognitive.microsoft.com')
    LUIS_API_ENDPOINT = os.environ.get("LUIS_API_ENDPOINT", 'https://chatbotflyme.cognitiveservices.azure.com/')
    
    #APP INSIGHTS CONFIGURATION 
    APPINSIGHTS_INSTRUMENTATION_KEY = os.environ.get("APPINSIGHTSINSTRUMENTATIONKEY", '07684d95-4649-4aad-9d19-4ed93838a246')
    