import requests


def query():
    # URL of the server (assuming it's running on localhost on port 8000)
    url = "http://localhost:8000"
    params = {"key1": "value1", "key2": "value2"}

    # Make a GET request to the server
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        print("Success!")
        print(response.text)  # Print the response text (HTML content in this case)
    else:
        print("An error has occurred.", response.status_code)


def main():
    query()


if __name__ == "__main__":
    main()
