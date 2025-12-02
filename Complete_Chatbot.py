import os
import json
import random
import re
import requests
import ast
import operator as op
from datetime import datetime, timedelta

# ---> ADD THESE TWO LINES <---
from amadeus import Client, ResponseError
# ---> END OF ADDITION <---

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import the necessary libraries
from groq import Groq


def _ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings or {}
        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def bag_of_words(self, words):
        bag = [0] * len(self.vocabulary)
        for s in words:
            for i, w in enumerate(self.vocabulary):
                if w == s:
                    bag[i] = 1
        return bag

    def parse_intents(self):
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)
        for intent in intents_data['intents']:
            tag = intent['tag']
            if tag not in self.intents:
                self.intents.append(tag)
            self.intents_responses[tag] = intent.get('responses', [])
            for pattern in intent.get('patterns', []):
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, tag))
        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags, indices = [], []
        for pattern_words, tag in self.documents:
            bag = self.bag_of_words(pattern_words)
            intent_index = self.intents.index(tag)
            bags.append(bag)
            indices.append(intent_index)
        self.X = np.array(bags, dtype=np.float32)
        self.y = np.array(indices, dtype=np.int64)

    def train_model(self, batch_size=8, lr=0.01, epochs=100):
        X_tensor, y_tensor = torch.tensor(self.X, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w', encoding='utf-8') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents), 'vocabulary': self.vocabulary, 'intents': self.intents}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r', encoding='utf-8') as f:
            dimensions = json.load(f)
        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.vocabulary = dimensions['vocabulary']
        self.intents = dimensions['intents']
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(bag_tensor)
        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]
        if predicted_intent in self.function_mappings:
            try:
                result = self.function_mappings[predicted_intent](input_message)
                base_response = random.choice(self.intents_responses.get(predicted_intent, ['']))
                return f"{base_response} {result}"
            except Exception as e:
                print(f"Function mapping error for '{predicted_intent}': {e}")
        responses = self.intents_responses.get(predicted_intent, [])
        return random.choice(responses) if responses else "I'm sorry, I don't understand that."
    

# --- All External Functions ---

# ---> COPY/PASTE THESE TWO FUNCTIONS FROM FLIGHT.PY <---

def get_city_code(amadeus, city_name):
    """Finds the IATA city code for a city name using Amadeus."""
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
    """Searches for flight offers using Amadeus."""
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
        # --- MODIFICATION ---
        # Was: print(error)
        # Now: return a dictionary with an error key
        print(f"Amadeus API Error: {error}")
        return {"error": f"Amadeus API Error: {error.description[0]['detail']}"}
        # --- END OF MODIFICATION ---

# ---> END OF COPY/PASTE <---



    # ... rest of your functions

# --- All External Functions ---
def get_joke(_msg=None):
    try:
        r = requests.get("https://v2.jokeapi.dev/joke/Any?type=single", timeout=5)
        r.raise_for_status()
        return r.json().get("joke", "Couldn't fetch a joke.")
    except Exception as e: return f"Error fetching joke: {e}"

def get_stocks(_msg=None):
    return ", ".join(random.sample(['AAPL', 'MSFT', 'GOOGL', 'AMZN'], 3))

def get_weather_from_message(message):
    match = re.search(r"in\s+([a-zA-Z\s]+)", message.lower())
    city = match.group(1).strip().title() if match else "London"
    API_KEY = os.getenv("OPENWEATHER_API_KEY")

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        return f"The weather in {city} is {data['weather'][0]['description']} with a temperature of {data['main']['temp']}°C."
    except Exception as e: return f"Sorry, I couldn't fetch the weather for {city} right now."

def get_news(_msg=None):
    API_KEY = os.getenv("NEWS_API_KEY")

    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEY}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        articles = r.json().get("articles", [])[:3]
        return " | ".join([a["title"] for a in articles]) or "No news found."
    except Exception as e: return f"Error fetching news: {e}"

operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}
def safe_eval(expr):
    try:
        node = ast.parse(expr, mode='eval').body
        def _eval(node):
            if isinstance(node, ast.Num): return node.n
            if isinstance(node, ast.BinOp): return operators[type(node.op)](_eval(node.left), _eval(node.right))
            raise TypeError("Unsupported")
        return _eval(node)
    except Exception: return "Invalid calculation."

def calculator(msg=None):
    expr = re.sub(r"(?i)calculate|solve|what is", "", msg or "").strip()
    return safe_eval(expr)

def get_city_coordinates(city_name):
    headers = {'User-Agent': 'TravelChatbot/1.0 (your.email@example.com)'}
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': city_name, 'format': 'json', 'limit': 1}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return (data[0]['lat'], data[0]['lon']) if data else (None, None)
    except Exception as e:
        print(f"Geocoding error for {city_name}: {e}")
        return None, None
    
def get_places_to_visit_osm(message):
    CATEGORY_MAPPING = { "beaches": ("natural", "beach"), "churches": ("amenity", "place_of_worship"), "museums": ("tourism", "museum"), "parks": ("leisure", "park"), "restaurants": ("amenity", "restaurant"), "cafes": ("amenity", "cafe"), "hotels": ("tourism", "hotel") }
    location_match = re.search(r"in\s+([a-zA-Z\s]+)", message.lower())
    if not location_match: return "Which city are you interested in? For example: 'beaches in Goa'"
    city = location_match.group(1).strip()
    category, (tag_key, tag_value) = "attractions", ("tourism", "attraction")
    for keyword, (key, value) in CATEGORY_MAPPING.items():
        if keyword in message.lower():
            category, tag_key, tag_value = keyword, key, value
            break
    lat, lon = get_city_coordinates(city)
    if not lat: return f"Sorry, I couldn't find the location for {city.title()}."
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""[out:json];(nwr["{tag_key}"="{tag_value}"](around:75000,{lat},{lon}););out center;"""
    try:
        headers = {'User-Agent': 'TravelChatbot/1.0 (your.email@example.com)'}
        r = requests.post(overpass_url, data=overpass_query, timeout=30, headers=headers)
        r.raise_for_status()
        places = r.json().get('elements', [])
        if not places: return f"I couldn't find any {category} in {city.title()}."
        place_names = {p['tags']['name'] for p in places if 'name' in p.get('tags', {})}
        if not place_names: return f"I found {category} in {city.title()}, but they weren't named."
        formatted_places = ", ".join(list(place_names)[:5])
        map_link = f"https://www.openstreetmap.org/#map=10/{lat},{lon}"
        return f"Here are some {category} in {city.title()}: {formatted_places}. Explore here: {map_link}"
    except Exception as e: return f"Map service error: {e}"

def plan_itinerary(message):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    except Exception as e:
        return "Itinerary service is not configured. Please add a Groq API key."
    
    city_match = re.search(r"to\s+([a-zA-Z\s]+)|for\s+([a-zA-Z\s]+)", message.lower())
    day_match = re.search(r"(\d+)\s*day", message.lower())
    
    if not city_match: return "Which city and for how many days?"
    
    city = (city_match.group(1) or city_match.group(2)).strip().title()
    days = day_match.group(1) if day_match else "3"
    user_details = message
    prompt = f"""Create a detailed, day-by-day travel itinerary based on the following user request: Request: \"{user_details}\". Details to extract and use: - Destination: {city} - Duration: {days} days. Please generate a logical and helpful itinerary. Format the output clearly with 'Day 1:', 'Day 2:', etc."""
    
    try:
        model_list = client.models.list().data
        selected_model = next((model.id for model in model_list if "whisper" not in model.id.lower() and "guard" not in model.id.lower()), None)

        if not selected_model:
            return "Sorry, no suitable AI models are available right now. Please try again later."
        
        print(f"--- Dynamically selected model: {selected_model} ---")

        chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=selected_model)
        itinerary = chat_completion.choices[0].message.content
        return f"\n{itinerary.strip()}"
    except Exception as e:
        print(f"Groq API call failed: {e}")
        return f"Sorry, I couldn't generate an itinerary for {city} right now."

def get_stay_options(message):
    location_match = re.search(r"in\s+([a-zA-Z\s]+)", message.lower())
    if not location_match: return "Which city are you looking for hotels in?"
    city = location_match.group(1).strip()
    lat, lon = get_city_coordinates(city)
    if not lat: return f"Sorry, I couldn't find the location for {city.title()}."
    tag_key, tag_value = "tourism", "hotel"
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""[out:json];(nwr["{tag_key}"="{tag_value}"](around:25000,{lat},{lon}););out center;"""
    try:
        headers = {'User-Agent': 'TravelChatbot/1.0 (your.email@example.com)'}
        r = requests.post(overpass_url, data=overpass_query, timeout=30, headers=headers)
        r.raise_for_status()
        places = r.json().get('elements', [])
        if not places: return f"I couldn't find any hotels listed for {city.title()}."
        hotel_names = {p['tags']['name'] for p in places if 'name' in p.get('tags', {})}
        if not hotel_names: return f"I found hotel locations in {city.title()}, but they weren't named."
        formatted_hotels = ", ".join(list(hotel_names)[:5])
        map_link = f"https://www.openstreetmap.org/#map=13/{lat},{lon}"
        return f"Some hotels in {city.title()} include: {formatted_hotels}. You can explore them here: {map_link}"
    except Exception as e: return f"Map service error: {e}"

# --- AVIATIONSTACK FLIGHT SEARCH FUNCTIONS (WITH ALL FIXES) ---

# ---> PASTE YOUR AVIATIONSTACK API KEY HERE <---

# ... (all your other functions)

# --- DELETE find_flights_aviationstack and get_iata_code_from_aviationstack ---
# --- ADD THIS NEW FUNCTION IN ITS PLACE ---

def find_flights_amadeus(message):
    """
    Parses a user's message for origin, destination, and date.
    If all are present, searches Amadeus for flights.
    If not, asks the user for the correct format.
    """
    
    # 1. Define the regex to find all three pieces of information
    # Example: "flights from Delhi to Guwahati on 2025-11-20"
    match = re.search(
        r"from\s+(.*?)\s+to\s+(.*?)\s+on\s+(\d{4}-\d{2}-\d{2})", 
        message, 
        re.IGNORECASE
    )

    # 2. Check if all info was found. If not, "ask" for it.
    if not match:
        # This is the "asking" part. We guide the user.
        return ("I can help with flights! Please provide the origin, "
                "destination, and date in this format:\n"
                "'flights from [Origin] to [Destination] on [YYYY-MM-DD]'")

    # 3. If we have a match, extract the data
    origin_city = match.group(1).strip().title()
    dest_city = match.group(2).strip().title()
    dep_date = match.group(3).strip()

    # 4. Initialize the Amadeus client (using your keys from flight.py)
    try:
       amadeus = Client(
    client_id=os.getenv("AMADEUS_CLIENT_ID"),
    client_secret=os.getenv("AMADEUS_CLIENT_SECRET")
)

    except Exception as e:
        print(f"Failed to initialize Amadeus client: {e}")
        return "Sorry, the flight search service is currently down."

    # 5. Get IATA codes
    dep_code = get_city_code(amadeus, origin_city)
    arr_code = get_city_code(amadeus, dest_city)

    if not dep_code:
        return f"Sorry, I couldn't find an airport code for '{origin_city}'."
    if not arr_code:
        return f"Sorry, I couldn't find an airport code for '{dest_city}'."

    print(f"Searching Amadeus: {origin_city} ({dep_code}) to {dest_city} ({arr_code}) on {dep_date}")

    # 6. Search for flights
    flights = search_flights(amadeus, dep_code, arr_code, dep_date)

    # 7. Format the response to be returned as a single string
    if not flights:
        return f"Sorry, no flights were found from {origin_city} to {dest_city} on {dep_date}."
    
    # Check if the search_flights function returned an error
    if isinstance(flights, dict) and 'error' in flights:
        return f"Sorry, I couldn't search for flights: {flights['error']}"

    # Build the success response string
    response_str = f"\nHere are the top flight offers from {origin_city} to {dest_city} on {dep_date}:\n"
    
    for offer in flights[:3]:  # Show top 3 offers
        price = offer['price']['total']
        itineraries = offer['itineraries'][0]['segments']
        
        response_str += f"\n--- Price: ₹{price} INR ---\n"
        
        # Check for one-way or connecting flights
        if len(itineraries) > 1:
            response_str += f"  (Connecting flight with {len(itineraries)} stops)\n"
        
        first_seg = itineraries[0]
        last_seg = itineraries[-1]
        
        response_str += (f"  Flight: {first_seg['carrierCode']}{first_seg['number']}\n"
                         f"  Departs: {first_seg['departure']['iataCode']} at {first_seg['departure']['at']}\n"
                         f"  Arrives: {last_seg['arrival']['iataCode']} at {last_seg['arrival']['at']}\n")

    return response_str

def handle_complex_trip(message):
    """
    Handles a request for BOTH flights and itinerary.
    Expected format: "...from [Origin] to [Dest] on [Date] for [N] days..."
    """
    
    # 1. Define a "Super Regex" to catch all 4 variables
    # Looks for: "from X to Y on Z ... for N days"
    match = re.search(
        r"from\s+(.*?)\s+to\s+(.*?)\s+on\s+(\d{4}-\d{2}-\d{2}).*?(\d+)\s*days", 
        message, 
        re.IGNORECASE
    )

    if not match:
        return ("To plan a complete trip, please use this format:\n"
                "'Plan trip from [Origin] to [Destination] on [YYYY-MM-DD] for [N] days'")

    # 2. Extract the data
    origin = match.group(1).strip()
    destination = match.group(2).strip()
    date = match.group(3).strip()
    duration = match.group(4).strip()

    response_text = f"Ok, planning a {duration}-day trip to {destination} flying from {origin} on {date}.\n"
    response_text += "="*40 + "\n"

    # 3. CALL FLIGHT FUNCTION (Reuse your existing logic!)
    # We construct a synthetic message to trick the flight function
    print("DEBUG: Calling Flight Module...")
    flight_query = f"flights from {origin} to {destination} on {date}"
    flight_result = find_flights_amadeus(flight_query)
    
    response_text += "✈️ **FLIGHT OPTIONS**\n"
    response_text += flight_result + "\n\n"

    # 4. CALL ITINERARY FUNCTION (Reuse your existing logic!)
    # We construct a synthetic message for the itinerary function
    print("DEBUG: Calling Itinerary Module...")
    itinerary_query = f"Plan a {duration} day itinerary for {destination}"
    
    # NOTE: I am assuming 'plan_itinerary' takes a message string as input. 
    # If it takes specific arguments, adjust this line.
    itinerary_result = plan_itinerary(itinerary_query) 
    
    response_text += "mei **ITINERARY PLAN**\n"
    response_text += itinerary_result + "\n"
    
    return response_text

# --- END OF NEW FUNCTION ---

# --- Main Execution ---
if __name__ == '__main__':
    _ensure_nltk()
    assistant = ChatbotAssistant(
        'intents.json',
        function_mappings={
            'stocks': get_stocks,
            'weather': get_weather_from_message,
            'news': get_news,
            'jokes': get_joke,
            'calculator': calculator,
            'travel_places_to_visit': get_places_to_visit_osm,
            'plan_itinerary': plan_itinerary,
            'find_hotels': get_stay_options,
            'find_flights': find_flights_amadeus,
            'plan_trip_with_flights':handle_complex_trip
        }
    )
    assistant.parse_intents()
    assistant.prepare_data()
    needs_retrain = True
    if os.path.exists('chatbot_model.pth') and os.path.exists('dimensions.json'):
        with open('dimensions.json', 'r') as f: saved_dims = json.load(f)
        if (saved_dims.get('input_size') == assistant.X.shape[1] and
            saved_dims.get('output_size') == len(assistant.intents)):
            needs_retrain = False
    if needs_retrain:
        print("Training new model...")
        assistant.train_model()
        assistant.save_model('chatbot_model.pth', 'dimensions.json')
    else:
        print("Loading existing model...")
        assistant.load_model('chatbot_model.pth', 'dimensions.json')
    print("\nChatbot is ready! Type '/quit' to exit.")
    while True:
        message = input("You: ")
        if message.lower() == '/quit': break
        print(f"Bot: {assistant.process_message(message)}")

 