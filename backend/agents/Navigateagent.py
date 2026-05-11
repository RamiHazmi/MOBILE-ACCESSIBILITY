"""
NavigateAgent.py — v5 (smart Overpass + multilingual + LangGraph)
─────────────────────────────────────────────────────────────────
v5 merges the proven standalone-script Overpass logic into the
production agent:

1. SMART OVERPASS LOOKUP
   • Lowercase + accent-strip normalization on the keyword
   • Includes node + way + RELATION (many buildings are
     multipolygons in OSM and were being missed before)
   • Collects ALL candidates with haversine distance, sorts,
     and prints the top 5 nearest for debugging — so you can
     verify what the agent actually picked vs the map.
   • out center tags  (gets coordinates AND the name in one
     query, no second lookup needed).

2. STAGED RADIUS SEARCH
   500 m → 1.5 km → 4 km → 12 km.  As soon as any POI with a
   matching tag is found we stop.  The previous 14-km-pharmacy
   problem cannot recur because the close-radius search runs first.

3. NOMINATIM FALLBACK
   Only used when Overpass returns nothing — viewbox-bounded
   around the user, then country-wide, then global.

4. EVERYTHING REMAINS 100% FREE
   • Overpass /interpreter   — free, please respect rate limits
   • Nominatim search        — free, 1 req/s
   • OSRM /route/v1          — free public demo
   No API keys, no credit cards.

FIX v5.1:
   • Overpass HTTP 406 — was caused by wrong Content-Type on POST.
     Now sends properly URL-encoded body with explicit header.
   • OSRM NoRoute — POIs returned by Overpass are often building
     centroids / polygon interiors that lie off the road graph.
     Added _snap_to_road() which calls OSRM /nearest/v1/ to snap
     both origin and destination onto the nearest routable node
     before calling /route/v1/.
"""

from __future__ import annotations

import math
import time
import unicodedata
import requests
from typing import Optional

# Localized phrases (so spoken nav follows the user's language)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phrasebook import t as _t


# ── Generic POI → OSM amenity/shop tag map ─────────────────────
# Used by Overpass to do tag-aware proximity search.
# All keys are normalized (lowercase, accents stripped) so lookups
# work regardless of how the user phrased the noun.
_POI_TAGS: dict[str, list[tuple[str, str]]] = {
    # Health / pharmacy
    "pharmacy":      [("amenity", "pharmacy")],
    "pharmacie":     [("amenity", "pharmacy")],
    "صيدلية":         [("amenity", "pharmacy")],
    "صيدلة":          [("amenity", "pharmacy")],
    "hospital":      [("amenity", "hospital")],
    "hopital":       [("amenity", "hospital")],
    "مستشفى":         [("amenity", "hospital")],
    "clinic":        [("amenity", "clinic")],
    "clinique":      [("amenity", "clinic")],
    "doctor":        [("amenity", "doctors")],
    "medecin":       [("amenity", "doctors")],
    "dentist":       [("amenity", "dentist")],
    "dentiste":      [("amenity", "dentist")],

    # Food / drink
    "restaurant":    [("amenity", "restaurant")],
    "مطعم":           [("amenity", "restaurant")],
    "cafe":          [("amenity", "cafe"), ("shop", "coffee")],
    "coffee":        [("amenity", "cafe"), ("shop", "coffee")],
    "coffee shop":   [("amenity", "cafe"), ("shop", "coffee")],
    "café":          [("amenity", "cafe"), ("shop", "coffee")],
    "مقهى":           [("amenity", "cafe"), ("shop", "coffee")],
    "قهوة":           [("amenity", "cafe"), ("shop", "coffee")],
    "fast food":     [("amenity", "fast_food")],
    "bakery":        [("shop", "bakery")],
    "boulangerie":   [("shop", "bakery")],
    "مخبزة":          [("shop", "bakery")],

    # Shops
    "supermarket":   [("shop", "supermarket"), ("shop", "convenience")],
    "supermarche":   [("shop", "supermarket"), ("shop", "convenience")],
    "supermarché":   [("shop", "supermarket")],
    "سوبر ماركت":     [("shop", "supermarket")],
    "convenience":   [("shop", "convenience")],
    "grocery":       [("shop", "supermarket"), ("shop", "convenience")],

    # Money
    "bank":          [("amenity", "bank")],
    "banque":        [("amenity", "bank")],
    "بنك":            [("amenity", "bank")],
    "atm":           [("amenity", "atm")],
    "distributeur":  [("amenity", "atm")],

    # Transport
    "gas station":   [("amenity", "fuel")],
    "fuel":          [("amenity", "fuel")],
    "station service": [("amenity", "fuel")],
    "محطة وقود":      [("amenity", "fuel")],
    "metro":         [("railway", "station"), ("railway", "subway_entrance")],
    "metro station": [("railway", "subway_entrance")],
    "train station": [("railway", "station")],
    "gare":          [("railway", "station")],
    "محطة":           [("railway", "station")],
    "bus stop":      [("highway", "bus_stop")],
    "arret bus":     [("highway", "bus_stop")],

    # Civic
    "post office":   [("amenity", "post_office")],
    "poste":         [("amenity", "post_office")],
    "بريد":           [("amenity", "post_office")],
    "police":        [("amenity", "police")],
    "library":       [("amenity", "library")],
    "bibliotheque":  [("amenity", "library")],
    "school":        [("amenity", "school")],
    "ecole":         [("amenity", "school")],
    "école":         [("amenity", "school")],
    "مدرسة":          [("amenity", "school")],
    "university":    [("amenity", "university")],
    "universite":    [("amenity", "university")],
    "جامعة":          [("amenity", "university")],
    "toilet":        [("amenity", "toilets")],
    "toilettes":     [("amenity", "toilets")],

    # Worship
    "mosque":        [("amenity", "place_of_worship")],
    "mosquee":       [("amenity", "place_of_worship")],
    "mosquée":       [("amenity", "place_of_worship")],
    "جامع":           [("amenity", "place_of_worship")],
    "مسجد":           [("amenity", "place_of_worship")],
    "church":        [("amenity", "place_of_worship")],
    "eglise":        [("amenity", "place_of_worship")],
    "église":        [("amenity", "place_of_worship")],
    "كنيسة":          [("amenity", "place_of_worship")],

    # Leisure
    "park":          [("leisure", "park")],
    "parc":          [("leisure", "park")],
    "حديقة":          [("leisure", "park")],
    "playground":    [("leisure", "playground")],
}


# ── Math / format helpers ──────────────────────────────────────

def _haversine(lat1, lng1, lat2, lng2) -> float:
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lng2 - lng1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _fmt_distance(metres: float) -> str:
    if metres >= 1000:
        return f"{metres / 1000:.1f} km"
    return f"{int(metres)} m"


def _fmt_duration(seconds: float) -> str:
    minutes = int(seconds / 60)
    if minutes < 1:
        return "less than a minute"
    if minutes < 60:
        return f"{minutes} minutes" if minutes > 1 else "one minute"
    hours = minutes // 60
    mins  = minutes % 60
    return f"{hours} hour {mins} min" if mins else f"{hours} hour"


def _is_latlng(s: str) -> bool:
    if "," not in s:
        return False
    return all(p.replace(".", "").replace("-", "").strip().isdigit()
               for p in s.split(","))


def _viewbox(lat: float, lng: float, radius_km: float = 3.0) -> str:
    delta_lat = radius_km / 111.0
    delta_lng = radius_km / (111.0 * math.cos(math.radians(lat)))
    return (f"{lng - delta_lng},{lat + delta_lat},"
            f"{lng + delta_lng},{lat - delta_lat}")


def _normalize_keyword(s: str) -> str:
    """Lowercase + strip Latin accents so 'Pharmacie' / 'pharmacië'
    both look up the same key as 'pharmacie'.  Arabic is left alone."""
    if not s:
        return ""
    s = s.strip().lower()
    nfkd = unicodedata.normalize("NFKD", s)
    out = "".join(c for c in nfkd
                  if not unicodedata.combining(c)
                  or "\u0600" <= c <= "\u06FF")
    return out


def _poi_tags_for(place: str) -> list[tuple[str, str]]:
    """Look up Overpass tags for a generic place word.  Tries an
    exact normalized match first, then a substring fallback."""
    key = _normalize_keyword(place)
    if not key:
        return []
    if key in _POI_TAGS:
        return _POI_TAGS[key]
    # Substring fallback — "find me a pharmacy please" → matches "pharmacy"
    for k, tags in _POI_TAGS.items():
        if k in key or key in k:
            return tags
    return []


# ── NavigateAgent ───────────────────────────────────────────────

class NavigateAgent:

    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
    OSRM_URL         = "https://router.project-osrm.org/route/v1"
    OSRM_NEAREST_URL = "https://router.project-osrm.org/nearest/v1"
    _OSRM_ROUTE_FALLBACKS = [
        ("foot", "https://routing.openstreetmap.de/routed-foot/route/v1"),
        ("car",  "https://routing.openstreetmap.de/routed-car/route/v1"),
    ]

    _NOMINATIM_INTERVAL = 1.1
    OFF_ROUTE_THRESHOLD = 80
    OFF_ROUTE_STRIKES   = 3

    # Staged radius search — 500 m first, then progressively wider.
    # Stops at the first radius that returns at least one hit.
    # 2000 m added to close the 1500→4000 m blind spot.
    OVERPASS_RADII_M = (500, 2000, 4000, 12000)

    # Fallback Overpass endpoint tried when the primary returns non-200.
    _OVERPASS_FALLBACK_URL = "https://overpass.kumi.systems/api/interpreter"

    def __init__(self, api_key: str = ""):
        self._last_nominatim_call = 0.0
        self._reset()

    def _reset(self):
        self.active          = False
        self.steps: list     = []
        self.step_latlngs    = []
        self.current_step    = 0
        self.destination_str = ""
        self.mode            = "walking"
        self.language        = "en"
        self.off_route_count = 0
        self.total_distance  = ""
        self.total_duration  = ""
        self._dest_lat: Optional[float] = None
        self._dest_lng: Optional[float] = None
        self._user_lat: Optional[float] = None
        self._user_lng: Optional[float] = None
        # Speech dedup — avoid repeating the same instruction every
        # GPS poll while the user is walking the same step.
        self._last_speak_key: Optional[str] = None
        self._last_speak_ts:  float          = 0.0
        # Min seconds between identical "in N m, turn right" lines
        self._speak_min_interval_s: float = 25.0

    @property
    def has_active_trip(self) -> bool:
        return self.active and len(self.steps) > 0

    # ── Road snapping ─────────────────────────────────────────
    # FIX: Overpass often returns building centroids / polygon
    # interiors that lie off the OSRM road graph, causing NoRoute.
    # Snapping both endpoints to the nearest routable node first
    # eliminates this problem.

    def _snap_to_road(self, lat: float, lng: float, mode: str) -> tuple[float, float]:
        """
        Snap a coordinate to the nearest point on the OSRM road graph.
        Calls /nearest/v1/<profile>/<lng>,<lat> with number=1.
        Falls back to the original coordinate if the request fails
        or the snapped point is more than 500 m away (likely wrong
        side of a motorway / ferry / etc.).
        """
        profile = "foot" if mode == "walking" else "car"
        url = f"{self.OSRM_NEAREST_URL}/{profile}/{lng},{lat}"
        try:
            r = requests.get(url, params={"number": 1}, timeout=8)
            data = r.json()
            if data.get("code") == "Ok" and data.get("waypoints"):
                loc = data["waypoints"][0]["location"]  # [lng, lat]
                snapped_lat, snapped_lng = loc[1], loc[0]
                dist = _haversine(lat, lng, snapped_lat, snapped_lng)
                # Sanity guard: if snap moved us >500 m, ignore it
                if dist > 500:
                    print(f"[OSRM/nearest] Snap too far ({int(dist)} m), "
                          f"keeping original {lat:.5f},{lng:.5f}")
                    return lat, lng
                print(f"[OSRM/nearest] {lat:.5f},{lng:.5f} → "
                      f"{snapped_lat:.5f},{snapped_lng:.5f} ({int(dist)} m)")
                return snapped_lat, snapped_lng
        except Exception as e:
            print(f"[OSRM/nearest] Snap error: {e}")
        return lat, lng  # fallback: original coords

    # ── Geocoding (Overpass first, Nominatim fallback) ───────

    def geocode(
        self,
        place: str,
        user_lat: Optional[float] = None,
        user_lng: Optional[float] = None,
    ) -> Optional[tuple[float, float, str]]:
        """
        Resolve a place name → (lat, lng, display_name).

        Strategy:
          1. Overpass POI search at staged radii (500 m → 12 km) if
             `place` matches a known generic POI word.  Returns the
             nearest hit by haversine distance.
          2. Nominatim viewbox-bounded search around user.
          3. Nominatim country-wide search (TN by default).
          4. Nominatim global search.

        Returns None if everything fails.
        """
        # ── 1. Overpass POI search (best for generic words) ──
        tags = _poi_tags_for(place)
        if tags and user_lat is not None and user_lng is not None:
            for i, radius_m in enumerate(self.OVERPASS_RADII_M):
                if i > 0:
                    time.sleep(0.4)   # avoid hammering the free Overpass API
                hit = self._overpass_nearest(user_lat, user_lng,
                                             tags, radius_m, place)
                if hit:
                    return hit
            # Overpass found nothing for a known category (pharmacy, cafe, …).
            # Do NOT fall through to Nominatim name-search: Nominatim would match
            # a venue literally *named* "Coffee Shop" or "Pharmacy X" anywhere in
            # the country, which is never what the user means.
            print(f"[NavigateAgent] '{place}' is a known POI category but "
                  f"0 hits within {self.OVERPASS_RADII_M[-1]}m — "
                  f"skipping Nominatim name-search to avoid wrong-country results.")
            return None

        # ── 2. Nominatim viewbox (proximity, BOUNDED) ────────
        # Only reached for specific named destinations (e.g. "Monoprix Lac",
        # "Hôpital Charles Nicolle") that are not in the generic POI tag map.
        if user_lat is not None and user_lng is not None:
            self._throttle_nominatim()
            try:
                params = {
                    "q":              place,
                    "format":         "json",
                    "limit":          10,
                    "addressdetails": 0,
                    "countrycodes":   "tn",
                    "viewbox":        _viewbox(user_lat, user_lng, 3.0),
                    "bounded":        1,   # hard-restrict to viewbox
                }
                r = requests.get(
                    self.NOMINATIM_URL, params=params,
                    headers={"User-Agent": "BlindGlassesApp/1.0"},
                    timeout=10)
                self._last_nominatim_call = time.time()
                data = r.json()
                if data:
                    best = min(data, key=lambda d: _haversine(
                        user_lat, user_lng,
                        float(d["lat"]), float(d["lon"])))
                    return (float(best["lat"]), float(best["lon"]),
                            best.get("display_name", place)[:80])
            except Exception as e:
                print(f"[NavigateAgent] Nominatim viewbox error: {e}")

        # ── 3. Nominatim country-wide ────────────────────────
        self._throttle_nominatim()
        try:
            params = {"q": place, "format": "json", "limit": 5,
                      "countrycodes": "tn"}
            r = requests.get(self.NOMINATIM_URL, params=params,
                             headers={"User-Agent": "BlindGlassesApp/1.0"},
                             timeout=10)
            self._last_nominatim_call = time.time()
            data = r.json()
            if data:
                if user_lat is not None and user_lng is not None:
                    best = min(data, key=lambda d: _haversine(
                        user_lat, user_lng,
                        float(d["lat"]), float(d["lon"])))
                else:
                    best = data[0]
                return (float(best["lat"]), float(best["lon"]),
                        best.get("display_name", place)[:80])
        except Exception as e:
            print(f"[NavigateAgent] Nominatim TN error: {e}")

        # ── 4. GLOBAL SEARCH REMOVED ─────────────────────────
        # Searching Nominatim globally for generic words like
        # 'coffee', 'cafe', 'pharmacy' returns results worldwide.
        # 'coffee' was matching a place in Georgia USA, causing
        # OSRM to try routing Tunisia→USA (NoRoute / 502).
        # If TN-bounded search found nothing → word is not a named
        # place in Tunisia. Tell user to be more specific.
        print(f"[NavigateAgent] '{place}' not found in Tunisia — "
              f"no global fallback (prevents cross-continent routing).")
        return None

    def _throttle_nominatim(self):
        elapsed = time.time() - self._last_nominatim_call
        if elapsed < self._NOMINATIM_INTERVAL:
            time.sleep(self._NOMINATIM_INTERVAL - elapsed)

    def _overpass_nearest(
        self, lat: float, lng: float,
        tags: list[tuple[str, str]], radius_m: int,
        original_query: str,
    ) -> Optional[tuple[float, float, str]]:
        """
        Query Overpass for POIs around (lat,lng) with the given tags.
        Returns the nearest hit (lat, lng, name).

        FIX: Previously used requests.post(data={"data": ql}) which
        sent multipart/form-data and triggered HTTP 406 from Overpass.
        Now sends a properly URL-encoded body with the correct
        Content-Type header that Overpass expects.

        Includes node + way + RELATION (multipolygon buildings
        like big supermarkets/hospitals were being missed before).
        Sorts ALL candidates by haversine distance and logs the top 5
        for debugging.
        """
        # Build OR-combined Overpass QL query
        parts = []
        for k, v in tags:
            parts.append(f'node["{k}"="{v}"](around:{radius_m},{lat},{lng});')
            parts.append(f'way["{k}"="{v}"](around:{radius_m},{lat},{lng});')
            parts.append(f'relation["{k}"="{v}"](around:{radius_m},{lat},{lng});')
        ql = f"[out:json][timeout:25];({''.join(parts)});out center tags;"

        try:
            # FIX (HTTP 406): Overpass requires the query to arrive as
            # an application/x-www-form-urlencoded POST body with a
            # "data" field.  requests.post(data=dict) sends
            # multipart/form-data which Overpass rejects with 406.
            encoded_body = "data=" + requests.utils.quote(ql)
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            # Try primary endpoint; fall back to mirror on any non-200.
            response = None
            for endpoint in (self.OVERPASS_URL, self._OVERPASS_FALLBACK_URL):
                try:
                    r = requests.post(endpoint, data=encoded_body,
                                      headers=headers, timeout=30)
                    if r.status_code == 200:
                        response = r
                        break
                    print(f"[Overpass] {endpoint} HTTP {r.status_code} "
                          f"('{original_query}' r={radius_m}m): {r.text[:80]}")
                except Exception as _ep:
                    print(f"[Overpass] {endpoint} error: {_ep}")

            if response is None:
                return None

            data = response.json()
            elems = data.get("elements", [])
            if not elems:
                print(f"[Overpass] '{original_query}' radius={radius_m}m → 0 hits")
                return None

            # Collect every candidate with its distance
            candidates = []
            for el in elems:
                if "lat" in el and "lon" in el:
                    elat, elng = el["lat"], el["lon"]
                elif "center" in el:
                    elat, elng = el["center"]["lat"], el["center"]["lon"]
                else:
                    continue
                d = _haversine(lat, lng, elat, elng)
                tags_d = el.get("tags", {}) or {}
                name = (tags_d.get("name")
                        or tags_d.get("brand")
                        or tags_d.get("operator")
                        or original_query)
                candidates.append((d, elat, elng, name))

            if not candidates:
                return None

            candidates.sort(key=lambda x: x[0])

            # Debug: print top 5 nearest so you can verify against the map
            print(f"[Overpass] '{original_query}' radius={radius_m}m "
                  f"→ {len(candidates)} hits, top 5:")
            for d, _, _, name in candidates[:5]:
                print(f"    {int(d):4d} m  {name}")

            d, elat, elng, name = candidates[0]
            return (elat, elng, name)
        except Exception as e:
            print(f"[Overpass] error: {e}")
            return None

    # ── OSRM route fetch ──────────────────────────────────────

    def _fetch_route(self, origin_lat, origin_lng,
                     dest_lat, dest_lng, mode):
        profile = "foot" if mode == "walking" else "car"
        coords  = f"{origin_lng},{origin_lat};{dest_lng},{dest_lat}"
        url     = f"{self.OSRM_URL}/{profile}/{coords}"
        try:
            r = requests.get(url, params={
                "steps": "true", "geometries": "geojson",
                "overview": "simplified", "annotations": "false",
            }, timeout=15)
            data = r.json()
            if data.get("code") != "Ok" or not data.get("routes"):
                print(f"[NavigateAgent] OSRM: {data.get('code')}")
                return None
            return data["routes"][0]
        except Exception as e:
            print(f"[NavigateAgent] OSRM error: {e}")
            return None

    @staticmethod
    def _parse_steps(route: dict, language: str = "en") -> list[dict]:
        steps = []
        for leg in route.get("legs", []):
            for s in leg.get("steps", []):
                man   = s.get("maneuver", {})
                m_t   = man.get("type", "")
                mod   = man.get("modifier", "")
                name  = s.get("name", "").strip()
                dist  = s.get("distance", 0.0)
                dur   = s.get("duration", 0.0)
                loc   = man.get("location", [0, 0])
                steps.append({
                    "instruction":  _osrm_instruction(m_t, mod, name, language),
                    "distance_m":   dist,
                    "distance_txt": _fmt_distance(dist),
                    "duration_txt": _fmt_duration(dur),
                    "end_lat":      loc[1],
                    "end_lng":      loc[0],
                })
        return [s for s in steps if s["distance_m"] > 1]

    # ── Public: start a new trip ──────────────────────────────

    # Tunisia bounding box — reject any origin that is clearly wrong.
    # If Flutter sends a stale cached position from outside Tunisia
    # (e.g. the Android default 37°N/-122°W or Georgia USA 31°N/-82°W)
    # OSRM will say NoRoute because it can't route across continents.
    _TN_LAT_MIN, _TN_LAT_MAX =  28.0,  38.5
    _TN_LNG_MIN, _TN_LNG_MAX =   7.0,  13.0

    def _bad_origin(self, lat, lng) -> bool:
        """True when the origin is clearly not in or near Tunisia."""
        if lat is None or lng is None:
            return True
        return not (self._TN_LAT_MIN <= lat <= self._TN_LAT_MAX
                    and self._TN_LNG_MIN <= lng <= self._TN_LNG_MAX)

    def start_trip(self, origin_lat, origin_lng, destination,
                   mode="walking", language="en") -> dict:
        # ── GPS sanity check ──────────────────────────────────
        # Reject stale / foreign coordinates before touching OSRM.
        # Common bad values: Android emulator default (37.4,-122.0),
        # Georgia USA (31.5,-82.8), or None when GPS has never locked.
        if self._bad_origin(origin_lat, origin_lng):
            msg = {
                "en": "GPS signal not ready. Please wait for a location fix.",
                "fr": "Signal GPS indisponible. Attendez la localisation.",
                "ar": "إشارة GPS غير متوفرة. انتظر لحظة.",
                "tn": "GPS ما جاش بعد. استنى شوية.",
            }.get((language or "en").lower()[:2],
                  "GPS signal not ready. Please wait for a location fix.")
            print(f"[NavigateAgent] Bad origin ({origin_lat}, {origin_lng}) "
                  f"— rejecting, outside Tunisia bounding box.")
            return self._err(msg)

        self._reset()
        self.mode            = mode
        self.language        = language
        self.destination_str = destination
        self._user_lat       = origin_lat
        self._user_lng       = origin_lng

        if _is_latlng(destination):
            parts    = destination.split(",")
            dest_lat = float(parts[0])
            dest_lng = float(parts[1])
            display  = destination
        else:
            res = self.geocode(destination,
                               user_lat=origin_lat, user_lng=origin_lng)
            if not res:
                return self._err(
                    f"I could not find '{destination}' near you. "
                    f"Try a more specific name.")
            dest_lat, dest_lng, display = res

        self._dest_lat = dest_lat
        self._dest_lng = dest_lng

        # Guard: Nominatim global fallback can return results outside
        # Tunisia. Reject any destination outside the bounding box.
        _dest_in_tn = (28.0 <= dest_lat <= 38.5) and (7.0 <= dest_lng <= 13.0)
        if not _dest_in_tn:
            print(f"[NavigateAgent] ✗ Destination ({dest_lat:.5f}, "
                  f"{dest_lng:.5f}) outside Tunisia — rejecting '{display}'.")
            return self._err(
                f"Could not find '{destination}' nearby. "
                f"Please try a more specific name.")

        # FIX (NoRoute): Snap both endpoints to the nearest routable
        # road node before calling OSRM.  Overpass / Nominatim often
        # return coordinates that are inside building polygons or
        # slightly off the road graph, which causes OSRM to return
        # NoRoute.  Snapping fixes this without changing the displayed
        # destination name or the straight-line distance calculation.
        dest_lat_r, dest_lng_r     = self._snap_to_road(dest_lat, dest_lng, mode)
        origin_lat_r, origin_lng_r = self._snap_to_road(origin_lat, origin_lng, mode)

        route = self._fetch_route(origin_lat_r, origin_lng_r,
                                  dest_lat_r, dest_lng_r, mode)
        if not route:
            return self._err("I could not get directions. Check your connection.")

        self.total_distance = _fmt_distance(route.get("distance", 0))
        self.total_duration = _fmt_duration(route.get("duration", 0))
        self.steps          = self._parse_steps(route, language=language)
        self.step_latlngs   = [(s["end_lat"], s["end_lng"]) for s in self.steps]

        if not self.steps:
            return self._err("Route has no walkable steps.")

        self.current_step = 0
        self.active       = True

        lang_norm = (language or "en").lower()
        mode_word = (_t("nav_mode_walk", lang_norm) if mode == "walking"
                     else _t("nav_mode_drive", lang_norm))
        first     = self.steps[0]["instruction"]
        first_d   = self.steps[0]["distance_txt"]
        straight  = _fmt_distance(_haversine(origin_lat, origin_lng,
                                             dest_lat, dest_lng))

        # Spoken-friendly destination — first 1-2 segments only.
        short_name = display.split(",")[0].strip()
        if not short_name:
            short_name = display
        cat = destination.strip().lower()
        if cat and cat not in short_name.lower() and len(cat) <= 20:
            short_name = f"{destination.strip().title()} on {short_name}"

        speak = _t("nav_found_route", lang_norm,
                   dest=short_name,
                   dist=self.total_distance,
                   dur=self.total_duration,
                   mode_word=mode_word,
                   first=first,
                   first_d=first_d)

        print(f"[NavigateAgent] {destination} → {display} "
              f"(straight {straight}, route {self.total_distance}, "
              f"{self.total_duration}, {len(self.steps)} steps)")

        return {
            "ok":            True,
            "speak":         speak,
            "duration_text": self.total_duration,
            "distance_text": self.total_distance,
            "destination":   display,
            "dest_lat":      dest_lat,
            "dest_lng":      dest_lng,
            "steps":         self.steps,
            "current_step":  0,
        }

    # ── Public: GPS update ────────────────────────────────────

    def update_position(self, lat, lng) -> dict:
        self._user_lat = lat
        self._user_lng = lng

        if not self.has_active_trip:
            return {"speak": "", "off_route": False, "arrived": False,
                    "detour_warning": None, "current_step": 0}

        lang = (self.language or "en").lower()
        step             = self.steps[self.current_step]
        end_lat, end_lng = self.step_latlngs[self.current_step]
        dist_to_end      = _haversine(lat, lng, end_lat, end_lng)

        # Arrived
        if self.current_step == len(self.steps) - 1 and dist_to_end < 25:
            self.active = False
            return {"speak": _t("nav_arrived", lang, dest=self.destination_str),
                    "off_route": False, "arrived": True,
                    "detour_warning": None, "current_step": self.current_step}

        # Advance to next step
        if dist_to_end < 20 and self.current_step < len(self.steps) - 1:
            self.current_step += 1
            step             = self.steps[self.current_step]
            end_lat, end_lng = self.step_latlngs[self.current_step]
            dist_to_end      = _haversine(lat, lng, end_lat, end_lng)
            speak = _t("nav_advance_step", lang,
                       instruction=step['instruction'],
                       dist=step['distance_txt'])
            self._last_speak_key = f"step{self.current_step}|advance"
            self._last_speak_ts  = time.time()
            return {"speak": speak, "off_route": False, "arrived": False,
                    "detour_warning": None,
                    "current_step": self.current_step,
                    "dist_to_next_m": int(dist_to_end)}

        # Off-route
        off_route      = self._check_off_route(lat, lng)
        detour_warning = None
        if off_route:
            self.off_route_count += 1
            if self.off_route_count >= self.OFF_ROUTE_STRIKES:
                detour_warning = _t("nav_off_route", lang)
                self.off_route_count = 0
        else:
            self.off_route_count = 0

        speak = ""
        if dist_to_end > 200:
            bucket = "far"
            candidate = _t("nav_continue_far", lang,
                           dist=_fmt_distance(dist_to_end))
        elif dist_to_end > 50:
            bucket = "approach"
            candidate = _t("nav_in_dist_then", lang,
                           dist=_fmt_distance(dist_to_end),
                           instruction=step['instruction'])
        else:
            bucket = "imminent"
            candidate = _t("nav_now", lang, instruction=step['instruction'])

        if detour_warning:
            speak = detour_warning
        else:
            key = f"step{self.current_step}|{bucket}"
            now = time.time()
            same_key = (key == self._last_speak_key)
            cooldown_passed = (now - self._last_speak_ts) >= self._speak_min_interval_s
            if (not same_key) or cooldown_passed or bucket == "imminent":
                speak = candidate
                self._last_speak_key = key
                self._last_speak_ts  = now

        return {"speak": speak, "off_route": off_route, "arrived": False,
                "detour_warning": detour_warning,
                "current_step": self.current_step,
                "dist_to_next_m": int(dist_to_end)}

    def recalculate(self, lat, lng) -> dict:
        return self.start_trip(origin_lat=lat, origin_lng=lng,
                               destination=self.destination_str,
                               mode=self.mode, language=self.language)

    def _check_off_route(self, lat, lng) -> bool:
        if not self.step_latlngs:
            return False
        end_lat, end_lng = self.step_latlngs[self.current_step]
        dist      = _haversine(lat, lng, end_lat, end_lng)
        step_dist = self.steps[self.current_step]["distance_m"]
        effective = max(self.OFF_ROUTE_THRESHOLD, step_dist * 0.5)
        return dist > effective + step_dist

    def _err(self, msg: str) -> dict:
        print(f"[NavigateAgent] {msg}")
        return {"ok": False, "speak": msg, "steps": [], "current_step": 0}


def _osrm_instruction(m_type, modifier, name, language: str = "en"):
    lang = (language or "en").lower()
    street_tpl = _t("nav_on_street", lang, street=name) if name else ""
    turn_map = {
        "left":         _t("nav_turn_left",   lang),
        "slight left":  _t("nav_slight_left", lang),
        "sharp left":   _t("nav_turn_left",   lang),
        "right":        _t("nav_turn_right",  lang),
        "slight right": _t("nav_slight_right",lang),
        "sharp right":  _t("nav_turn_right",  lang),
        "straight":     _t("nav_straight",    lang),
        "uturn":        _t("nav_uturn",       lang),
    }
    cont = _t("nav_straight", lang)
    if m_type == "depart":
        head = {"en": f"Head {modifier or 'forward'}",
                "fr": f"Allez {modifier or 'tout droit'}",
                "ar": f"اتجه {modifier or 'للأمام'}",
                "tn": f"امشي {modifier or 'قدام'}"}.get(lang,
                  f"Head {modifier or 'forward'}")
        return f"{head}{street_tpl}"
    if m_type == "arrive":
        return _t("nav_arrived", lang, dest="")
    if m_type in ("turn", "end of road"):
        return f"{turn_map.get(modifier, cont)}{street_tpl}"
    if m_type == "roundabout":
        rb = {"en": "Enter the roundabout and take the exit",
              "fr": "Entrez dans le rond-point et prenez la sortie",
              "ar": "ادخل الدوار واخرج عند المخرج",
              "tn": "ادخل الدوار واخرج"}.get(lang, "Enter roundabout")
        return f"{rb}{street_tpl}"
    if m_type in ("merge", "on ramp"):
        mg = {"en": "Merge", "fr": "Insérez-vous",
              "ar": "اندمج", "tn": "ادخل"}.get(lang, "Merge")
        return f"{mg}{street_tpl}"
    if m_type == "off ramp":
        ex = {"en": "Take the exit", "fr": "Prenez la sortie",
              "ar": "خذ المخرج",     "tn": "اخرج"}.get(lang, "Take the exit")
        return f"{ex}{street_tpl}"
    if m_type == "fork":
        keep = {"en": "Keep", "fr": "Restez",
                "ar": "ابق",   "tn": "ابقى"}.get(lang, "Keep")
        at_fork = {"en": " at the fork", "fr": " à la bifurcation",
                   "ar": " عند المفترق", "tn": " عند المفترق"}.get(lang, " at the fork")
        return f"{turn_map.get(modifier, keep)}{at_fork}{street_tpl}"
    return f"{turn_map.get(modifier, cont)}{street_tpl}"