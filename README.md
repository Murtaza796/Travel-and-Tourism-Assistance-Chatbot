🚀 Features
🗺️ Travel Itinerary Generator (AI-Powered)

Uses the Groq LLM API to generate custom multi-day itineraries:

“Plan a 3-day trip to Paris”

“Create a 5-day itinerary for Tokyo”

✈️ Flight Search

Supports two APIs:

AviationStack API — airport lookup + flight schedules

(Optional) Amadeus API — flight offers (if enabled)

Example queries:

“Find flights from Delhi to Guwahati”

“Show flights from Paris to New York”

🏨 Hotel Finder

Using OpenStreetMap Overpass API, the bot can find hotels around any city:

“Find hotels in Goa”

“Where can I stay in Paris?”

🧭 Places to Visit

Uses OSM Overpass API to find:

Beaches

Museums

Parks

Restaurants

Churches

Tourist attractions

Example:

“List museums in London”

“Beaches in Goa”

🌤️ Weather Information

Fetches real-time weather from the OpenWeather API.

Example:

“How’s the weather in Mumbai?”

📰 Latest News

Uses NewsAPI to get top headlines.

😂 Jokes, 📈 Stocks & 🔢 Calculator

Extras to make the chatbot more useful and fun:

Random programming jokes

Fake stock suggestions

A safe arithmetic calculator

🧠 ML-Based Intent Classification

Built using:

PyTorch

NLTK

Bag-of-Words model

Chatbot understands intent categories like:

greeting

weather

plan itinerary

find hotels

find flights

places to visit

jokes

calculator

Training data is stored in intents.json.







├── Complete_Chatbot.py         # Full chatbot version (multi-feature)
├── integrated.py               # Integrated and improved chatbot engine
├── flight.py                   # Standalone Amadeus flight lookup
├── intents.json                # Training data for ML-based intent model
├── chatbot_model.pth           # Trained PyTorch model (auto-generated)
├── dimensions.json             # Model metadata (auto-generated)
└── README.md                   # Documentation





🛠️ Tech Stack
Backend

Python 3

PyTorch

NLTK

Requests

Groq API

AviationStack API

OpenStreetMap Overpass API

OpenWeather API

NewsAPI

ML / NLP

Neural network for intent classification

Bag-of-words text preprocessing

Tokenization & lemmatization using NLTK
