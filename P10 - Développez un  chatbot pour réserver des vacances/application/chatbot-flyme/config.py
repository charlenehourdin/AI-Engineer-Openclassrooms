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
    APP_ID =os.environ.get('APP_ID', '')
    APP_PASSWORD = os.environ.get('APP_PASSWORD', '')
    
    #LUIS APP CONFIGURATION
    LUIS_APP_ID = os.environ.get("LUIS_APP_ID", '')
    LUIS_API_KEY = os.environ.get("LUIS_API_KEY", '')
    LUIS_API_HOST_NAME = os.environ.get("LUIS_API_HOST_NAME", '')
    LUIS_API_ENDPOINT = os.environ.get("LUIS_API_ENDPOINT", '')
    
    #APP INSIGHTS CONFIGURATION 
    APPINSIGHTS_INSTRUMENTATION_KEY = os.environ.get("APPINSIGHTSINSTRUMENTATIONKEY", '')
    
