#!/bin/bash

# Define the URL of the server
url="http://localhost:8000"
params="?param1=value1&param2=value2"

# Use curl to send a GET request
response=$(curl -s "${url}${params}") 

# Print the response
echo "Response from server:"
echo "$response"
