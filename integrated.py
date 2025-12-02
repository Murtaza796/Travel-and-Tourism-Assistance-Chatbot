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
    API_KEY = "YOUR_OPENWEATHER_API_KEY"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        return f"The weather in {city} is {data['weather'][0]['description']} with a temperature of {data['main']['temp']}°C."
    except Exception as e: return f"Sorry, I couldn't fetch the weather for {city} right now."

def get_news(_msg=None):
    API_KEY = "YOUR_NEWSAPI_KEY"
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
AVIATIONSTACK_API_KEY = "3d97fe5fa43a4bd624a8b1a4d418ff34" 

def get_iata_code_from_aviationstack(city_name):
    if not AVIATIONSTACK_API_KEY or AVIATIONSTACK_API_KEY == "YOUR_AVIATIONSTACK_API_KEY":
        print("AviationStack API key is missing.")
        return None
    params = { 'access_key': AVIATIONSTACK_API_KEY, 'search': city_name }
    try:
        response = requests.get('http://api.aviationstack.com/v1/airports', params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            print(f"AviationStack API Error (IATA): {data['error'].get('info')}")
            return None
        if data.get('data'):
            first_valid = next((apt['iata_code'] for apt in data['data'] if apt.get('iata_code')), None)
            return first_valid
        return None
    except Exception as e:
        print(f"AviationStack IATA lookup failed for {city_name}: {e}")
        return None

def find_flights_aviationstack(message):
    if not AVIATIONSTACK_API_KEY or AVIATIONSTACK_API_KEY == "3d97fe5fa43a4bd624a8b1a4d418ff34":
        return "The flight search service is not configured. Please add an AviationStack API key."

    # --- THIS IS THE NEW, SMARTER LOGIC ---
    cleaned_message = message.lower().replace("flights", "").replace("flight", "").strip()
    
    # Try to find the "from ... to ..." pattern first
    flight_match = re.search(r"from\s+(.*?)\s+to\s+(.*)", cleaned_message)
    if not flight_match:
        # If that fails, try the simpler "origin to destination" pattern
        flight_match = re.search(r"(.*?)\s+to\s+(.*)", cleaned_message)

    if not flight_match:
        return "Please specify your origin and destination, like 'flights from Delhi to Leh'."

    origin_city = flight_match.group(1).strip()
    dest_raw = flight_match.group(2).strip()
    
    dest_city = re.sub(r"(\?|what's the price|find flights|search for).*", "", dest_raw, flags=re.IGNORECASE).strip()
    # --- END OF FIX ---

    origin_iata = get_iata_code_from_aviationstack(origin_city)
    dest_iata = get_iata_code_from_aviationstack(dest_city)

    if not origin_iata:
        return f"Sorry, I couldn't find an airport code for '{origin_city.title()}'."
    if not dest_iata:
        return f"Sorry, I couldn't find an airport code for '{dest_city.title()}'."

    params = {
        'access_key': AVIATIONSTACK_API_KEY,
        'dep_iata': origin_iata,
        'arr_iata': dest_iata,
        'limit': 3
    }
    
    try:
        api_result = requests.get('http://api.aviationstack.com/v1/flights', params=params, timeout=15)
        api_result.raise_for_status()
        flight_data = api_result.json()

        if 'error' in flight_data:
            print(f"AviationStack API Error (Flights): {flight_data['error'].get('info')}")
            return "Sorry, there was an error searching for flights."

        if not flight_data or not flight_data.get('data'):
            return f"Sorry, I couldn't find any scheduled flights from {origin_iata} to {dest_iata}."

        output = f"\nHere are some scheduled flights from {origin_iata} to {dest_iata} (times are UTC):\n"
        for flight in flight_data['data']:
            airline = flight.get('airline', {}).get('name', 'Unknown Airline')
            flight_num = flight.get('flight', {}).get('iata', 'N/A')
            dep_time = flight.get('departure', {}).get('scheduled', 'N/A')
            arr_time = flight.get('arrival', {}).get('scheduled', 'N/A')
            
            output += f"- {airline} ({flight_num}), Departs: {dep_time}, Arrives: {arr_time}\n"
        
        output += "(Note: Free tier shows schedules, not prices.)"
        return output
        
    except Exception as e:
        print(f"AviationStack API call failed: {e}")
        return "Sorry, I'm having trouble searching for flight schedules."

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
            'find_flights': find_flights_aviationstack
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

 