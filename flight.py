from amadeus import Client, ResponseError

def get_city_code(amadeus, city_name):
    try:
        response = amadeus.reference_data.locations.get(
            keyword=city_name,
            subType='CITY'
        )
        # Return the first matching city code
        return response.data[0]['iataCode']
    except Exception as e:
        print(f"Error finding city code for {city_name}: {e}")
        return None

def search_flights(amadeus, origin, destination, date):
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=date,
            adults=1,
            currencyCode='INR'
        )
        return response.data
    except ResponseError as error:
        print(error)
        return []

def main():
    # Replace with your Amadeus API credentials
    amadeus = Client(
    client_id=os.getenv("AMADEUS_CLIENT_ID"),
    client_secret=os.getenv("AMADEUS_CLIENT_SECRET")
)


    print("Welcome to the Amadeus City-to-City Flight Finder!")
    dep_city = input("Enter Departure City Name: ")
    arr_city = input("Enter Arrival City Name: ")
    dep_date = input("Enter Departure Date (YYYY-MM-DD): ")

    dep_code = get_city_code(amadeus, dep_city)
    arr_code = get_city_code(amadeus, arr_city)

    if not dep_code or not arr_code:
        print("Could not find city codes for one or both cities.")
        return

    print(f"Searching flights from {dep_city} ({dep_code}) to {arr_city} ({arr_code}) on {dep_date}...")

    flights = search_flights(amadeus, dep_code, arr_code, dep_date)
    if not flights:
        print("No flights found for your query.")
    else:
        for offer in flights[:5]:  # Show top 5 offers
            itineraries = offer['itineraries'][0]['segments']
            price = offer['price']['total']
            print(f"Price: ₹{price} INR")
            for seg in itineraries:
                print(f"Flight: {seg['carrierCode']}{seg['number']}")
                print(f"From: {seg['departure']['iataCode']} at {seg['departure']['at']}")
                print(f"To: {seg['arrival']['iataCode']} at {seg['arrival']['at']}")
            print("-" * 40)

if __name__ == '__main__':
    main()