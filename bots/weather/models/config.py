CITY_COORDS: dict[str, tuple[float, float]] = {
    "NYC": (40.7772, -73.8726),
    "New York": (40.7772, -73.8726),
    "London": (51.5048, 0.0495),
    "Hong Kong": (22.3080, 113.9185),
    "Miami": (25.7953, -80.2902),
    "Dallas": (32.8471, -96.8518),
    "Atlanta": (33.6407, -84.4277),
    "Seoul": (37.4602, 126.4407),
    "Tokyo": (35.5522, 139.7796),
    "Los Angeles": (33.9425, -118.4081),
    "Chicago": (41.9742, -87.9073),
    "Paris": (48.9694, 2.4414),
    "Shanghai": (31.1443, 121.8083),
    "Buenos Aires": (-34.8222, -58.5358),
    "Jeddah": (21.6796, 39.1565),
    "Ankara": (40.1281, 32.9951),
    "Lagos": (6.5774, 3.3210),
    "São Paulo": (-23.4356, -46.4731),
    "Warsaw": (52.1657, 20.9671),
    "Taipei": (25.0796, 121.2340),
    "Austin": (30.1945, -97.6699),
    "Helsinki": (60.3172, 24.9635),
    "Denver": (39.8617, -104.6732),
    # --- New high-volume Polymarket cities ---
    "Beijing": (40.0799, 116.6031),
    "Madrid": (40.4722, -3.5611),
    "Wellington": (-41.3295, 174.8062),
    "Milan": (45.6296, 8.7264),
    "Wuhan": (30.7838, 114.2081),
    "Munich": (48.3537, 11.7860),
    "Moscow": (55.9731, 37.4146),
    "San Francisco": (37.6213, -122.3790),
    "Istanbul": (40.9769, 28.8140),
    "Jakarta": (-6.1274, 106.6537),
    "Mexico City": (19.4361, -99.0719),
    "Amsterdam": (52.3082, 4.7642),
    "Busan": (35.1797, 129.0750),
    "Seattle": (47.4435, -122.3017),
    "Toronto": (43.6778, -79.6248),
    "Cape Town": (-33.9716, 18.6021),
}

CITY_ALIASES: dict[str, str] = {
    "NYC": "New York",
    "LA": "Los Angeles",
    "HK": "Hong Kong",
    "SP": "São Paulo",
    "SAO": "São Paulo",
    "BSAS": "Buenos Aires",
    "BA": "Buenos Aires",
    "SF": "San Francisco",
    "CDMX": "Mexico City",
    "MEX": "Mexico City",
}


def resolve_city_alias(name: str) -> str:
    return CITY_ALIASES.get(name, name)


DEFAULT_CITIES = [
    "NYC",
    "London",
    "Tokyo",
    "Chicago",
    "Miami",
    "Seoul",
    "Warsaw",
    "Lagos",
    "Hong Kong",
    "Shanghai",
    "Beijing",
    "Madrid",
    "Taipei",
    "Paris",
    "Denver",
    "Wellington",
    "Milan",
    "Wuhan",
    "Munich",
    "Moscow",
    "San Francisco",
    "Istanbul",
    "Jakarta",
    "Mexico City",
    "Amsterdam",
    "Busan",
    "Seattle",
    "Toronto",
    "Cape Town",
    "Atlanta",
    "Dallas",
    "Austin",
    "Helsinki",
]

# Strategy defaults removed 2026-05-16 - all strategies unprofitable
# Framework preserved for future implementation with better forecasting models
# See: .trellis/spec/backend/trading-config.md for strategy design history

CACHE_TTL: dict[str, int] = {
    "markets": 300,
    "prices": 30,
    "forecast": 3600,
    "tags": 86400,
}

# Per-strategy default overrides, keyed by Strategy.name.
# Empty = use the dataclass defaults in each strategy's Config.
STRATEGY_DEFAULTS: dict[str, dict[str, float]] = {}
