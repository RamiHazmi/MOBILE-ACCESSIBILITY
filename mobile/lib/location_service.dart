import 'package:geolocator/geolocator.dart';

/// Tunisia bounding box — any cached position outside this is rejected.
/// This prevents stale Android defaults (37°N/-122°W) or foreign cached
/// positions (Georgia USA 31°N/-82°W) from being sent to the backend,
/// which would cause OSRM to return NoRoute across continents.
const double _latMin =  28.0;
const double _latMax =  38.5;
const double _lngMin =   7.0;
const double _lngMax =  13.0;

bool _isInTunisia(Position p) =>
    p.latitude  >= _latMin && p.latitude  <= _latMax &&
    p.longitude >= _lngMin && p.longitude <= _lngMax;

class LocationService {
  static Position? _last;

  /// Returns the current GPS position, or null if unavailable.
  ///
  /// - Requests permission if not granted.
  /// - Times out after 10 s and falls back to cached position.
  /// - ONLY returns a cached position if it is inside Tunisia.
  ///   A cached position from outside Tunisia (stale, emulator default,
  ///   foreign trip) is treated as null — the caller must wait.
  static Future<Position?> current() async {
    try {
      LocationPermission perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.denied) {
        perm = await Geolocator.requestPermission();
      }
      if (perm == LocationPermission.deniedForever) return null;

      _last = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
        timeLimit: const Duration(seconds: 10),
      );

      // Fresh fix — accept regardless of location (user might be abroad)
      return _last;

    } catch (_) {
      // GPS timed out — only return cached if it is in Tunisia.
      // If the cache is a foreign/stale position, return null so the
      // UI can tell the user to wait rather than route to the wrong place.
      final cached = _last;
      if (cached != null && _isInTunisia(cached)) {
        return cached;
      }
      // Cache is absent, foreign, or from a previous session abroad.
      return null;
    }
  }

  static Position? get cached => _last;
}