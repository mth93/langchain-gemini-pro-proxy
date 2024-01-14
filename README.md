## OAI API to Gemini Pro Middleware

## Overview

This API provides an OAI API compatible middleware for gemini-pro api.

## Installation

    Clone this repository:
    Bash

    git clone https://github.com/mth93/langchain-gemini-pro-proxy.git

Install the required dependencies:
Bash

cd langchain-gemini-pro-proxy
pip install -r requirements.txt


## Configuration

    Create a .env file in the project directory and set the following variables:
        OPENAI_API_KEY: Your Google API key with access to the relevant AI services.
        GEMINI_API_PORT (optional): The port on which to run the API (defaults to 8000).

## Usage

    Start the API server:
    Bash

    uvicorn app:app --reload

    Access the endpoints using a REST client or tools like Postman or openai library.

## Usage with other github projects

    set OPENAI_BASE_URL enviroment variable to the middleware url and port and set OPENAI_API_KEY to gemini-pro api key

## Available Endpoints

1. Text Completion
    URL:/v1/completions

2. Chat Completion
    URL:/v1/chat/completions

3. Embeddings
    URL:/v1/embeddings

for testing curl requests examples:
https://platform.openai.com/docs/api-reference 

## Additional Notes

    The API uses the models/embedding-001 model for embedding generation. Adjust this if needed.
    Logging is configured to write to a file named api_logs.log in the repository directory