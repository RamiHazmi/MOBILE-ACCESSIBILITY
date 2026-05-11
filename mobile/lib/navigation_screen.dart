// navigation_screen.dart — v4 (with embedded YOLO danger camera)
//
// What changed vs v3:
//
// 1. EMBEDDED CAMERA TILE
//    A small camera preview sits in the top-right corner.  Frames
//    are streamed at ~1 / 1.5 s to the backend /nav_danger endpoint
//    (which runs the user's fine-tuned YOLO best.pt).  Only NEAR
//    obstacles in the user's path produce a spoken warning — parked
//    cars and far objects stay silent.  Bounding boxes are drawn
//    over the small preview when a hazard is detected.
//
// 2. CALLS /reset ON CLOSE
//    Cleans the agent's state and YOLO cooldowns so the next
//    navigation session starts fresh.
//
// 3. GUARANTEED FREE STACK
//    OSM tiles, OSRM routing, Nominatim geocoding, on-device TTS,
//    on-device YOLO inference — no API key for any of them.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;
import 'package:latlong2/latlong.dart';
import 'package:vibration/vibration.dart';

import 'api_service.dart';
import 'location_service.dart';
import 'tts_service.dart';

// ─── Bounding box from /nav_danger response ───────────────────

class _NavDangerBox {
  final String label;
  final String proximity;
  final String direction;
  final double cx, cy, w, h;
  const _NavDangerBox({
    required this.label,
    required this.proximity,
    required this.direction,
    required this.cx,
    required this.cy,
    required this.w,
    required this.h,
  });
}

// ─────────────────────────────────────────────────────────────

class NavigationScreen extends StatefulWidget {
  final String destination;
  final String navMode;
  final String language;

  const NavigationScreen({
    Key? key,
    required this.destination,
    required this.navMode,
    required this.language,
  }) : super(key: key);

  @override
  State<NavigationScreen> createState() => _NavigationScreenState();
}

class _NavigationScreenState extends State<NavigationScreen> {

  // ── Endpoints (free) ──────────────────────────────────────
  static const String _osrmUrl = 'https://router.project-osrm.org/route/v1';
  static const String _nominatimUrl =
      'https://nominatim.openstreetmap.org/search';

  static const _gpsInterval    = Duration(seconds: 4);
  static const _dangerInterval = Duration(milliseconds: 1500);

  // ── Map state ─────────────────────────────────────────────
  final MapController _mapController = MapController();
  bool _mapReady = false;

  LatLng? _userLatLng;
  List<LatLng> _routePoints = [];
  LatLng? _destLatLng;

  // ── Nav state ─────────────────────────────────────────────
  String _statusText  = 'Getting your location…';
  String _etaText     = '';
  String _lastSpoken  = '';
  bool   _arrived     = false;
  bool   _offRoute    = false;
  bool   _disposed    = false;
  bool   _started     = false;
  bool   _requestBusy = false;
  int    _currentStep = 0;

  Timer? _gpsTimer;

  // ── Embedded camera state ─────────────────────────────────
  CameraController? _camCtrl;
  bool _cameraReady     = false;
  bool _cameraSendBusy  = false;
  Timer? _dangerTimer;
  String _dangerStatus  = 'LOW';
  String _lastDangerSpoken = '';
  List<_NavDangerBox> _dangerBoxes = [];
  bool _yoloReady = true;     // assume yes until backend says otherwise

  // ── Colours ───────────────────────────────────────────────
  static const _cyan  = Color(0xFF00B4D8);
  static const _red   = Color(0xFFFF3B3B);
  static const _amber = Color(0xFFFF9500);
  static const _green = Color(0xFF34C759);
  static const _bg    = Color(0xFF080810);

  Color get _barColor =>
      _offRoute ? _red
      : _arrived ? _green
      : (_dangerStatus == 'HIGH' ? _red
      : (_dangerStatus == 'MEDIUM' ? _amber : _cyan));

  // ═══════════════════════════════════════════════════════════
  // LIFECYCLE
  // ═══════════════════════════════════════════════════════════

  @override
  void initState() {
    super.initState();
    _startTrip();
    _initEmbeddedCamera();
  }

  @override
  void dispose() {
    _disposed = true;
    _gpsTimer?.cancel();
    _dangerTimer?.cancel();
    final cc = _camCtrl; _camCtrl = null;
    Future.delayed(const Duration(milliseconds: 300), () async {
      try { await cc?.dispose(); } catch (_) {}
    });
    try { _mapController.dispose(); } catch (_) {}
    // Tell backend to clear pending state + YOLO cooldowns
    ApiService.reset();
    super.dispose();
  }

  void _ss(VoidCallback fn) {
    if (!_disposed && mounted) setState(fn);
  }

  // ═══════════════════════════════════════════════════════════
  // TRIP LOGIC
  // ═══════════════════════════════════════════════════════════

  Future<void> _startTrip() async {
    final pos = await LocationService.current();
    if (pos == null) {
      _ss(() => _statusText =
          'GPS unavailable. Please enable location and try again.');
      return;
    }

    _ss(() {
      _userLatLng = LatLng(pos.latitude, pos.longitude);
      _statusText = 'Calculating route…';
    });

    await _fetchOsrmPolyline(pos.latitude, pos.longitude);

    if (widget.destination.isNotEmpty) {
      final result = await _callNavigate(
          lat: pos.latitude, lng: pos.longitude,
          dest: widget.destination);

      if (result == null) {
        _ss(() => _statusText = 'Could not get directions. Check connection.');
        return;
      }

      final speak = (result['speak'] ?? '') as String;
      if (speak.isNotEmpty && speak != _lastSpoken) {
        _lastSpoken = speak;
        TtsService.speak(speak, lang: widget.language);
      }
      _ss(() {
        if (speak.isNotEmpty) _statusText = speak;
        _started = true;
      });

      _gpsTimer = Timer.periodic(_gpsInterval, (_) => _updatePosition());
    } else {
      _ss(() => _statusText = 'Showing your current location');
      TtsService.speak(_statusText, lang: widget.language);
    }
  }

  Future<void> _updatePosition() async {
    if (_disposed || _arrived || _requestBusy) return;

    final pos = await LocationService.current();
    if (pos == null) return;

    final userLl = LatLng(pos.latitude, pos.longitude);
    _ss(() => _userLatLng = userLl);

    if (_mapReady) {
      try { _mapController.move(userLl, _mapController.camera.zoom); }
      catch (_) {}
    }

    _requestBusy = true;
    try {
      final result = await _callNavigate(
          lat: pos.latitude, lng: pos.longitude, dest: '');
      if (result == null || _disposed) return;

      final speak         = (result['speak'] ?? '') as String;
      final arrived       = result['arrived'] == true;
      final offRoute      = result['off_route'] == true;
      final detourWarning = result['detour_warning'] as String?;
      final currentStep   = (result['current_step'] ?? 0) as int;
      final distToNext    = result['dist_to_next_m'] as int?;
      final interrupt     = result['interrupt'] == true;

      _ss(() {
        _currentStep = currentStep;
        _offRoute    = offRoute;
        _arrived     = arrived;
        if (speak.isNotEmpty) _statusText = speak;
        if (distToNext != null) _etaText = '$distToNext m to next turn';
      });

      // Speak only when text actually changes — backend already
      // dedups, this is a second safety net for the UI.
      if (speak.isNotEmpty && speak != _lastSpoken) {
        _lastSpoken = speak;
        TtsService.speak(speak, lang: widget.language,
            interrupt: interrupt || detourWarning != null);
      }

      if (detourWarning != null || offRoute) _vibrate(strong: true);

      if (arrived) {
        _vibrate(strong: false);
        _gpsTimer?.cancel();
        _dangerTimer?.cancel();
        await Future<void>.delayed(const Duration(seconds: 3));
        if (!_disposed && mounted) Navigator.pop(context);
      }
    } finally {
      _requestBusy = false;
    }
  }

  // ═══════════════════════════════════════════════════════════
  // EMBEDDED CAMERA → /nav_danger
  // ═══════════════════════════════════════════════════════════

  Future<void> _initEmbeddedCamera() async {
    List<CameraDescription> cameras;
    try {
      cameras = await availableCameras();
    } catch (e) {
      print('[Nav] no cameras: $e');
      return;
    }
    if (cameras.isEmpty) return;

    final ctrl = CameraController(
      cameras.first, ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    try {
      await ctrl.initialize();
      await ctrl.setFlashMode(FlashMode.off);
      await ctrl.setFocusMode(FocusMode.auto);
      await ctrl.setExposureMode(ExposureMode.auto);
      await Future.delayed(const Duration(milliseconds: 800));
    } catch (e) {
      print('[Nav] camera init: $e');
      try { await ctrl.dispose(); } catch (_) {}
      return;
    }

    if (_disposed || !mounted) {
      try { await ctrl.dispose(); } catch (_) {}
      return;
    }

    _camCtrl = ctrl;
    _ss(() => _cameraReady = true);

    _dangerTimer = Timer.periodic(_dangerInterval, (_) => _scanDangerOnce());
  }

  Future<void> _scanDangerOnce() async {
    if (_disposed || _arrived) return;
    if (_cameraSendBusy) return;
    if (!_cameraReady || _camCtrl == null) return;
    final cc = _camCtrl!;
    if (!cc.value.isInitialized) return;

    _cameraSendBusy = true;
    try {
      final XFile     photo = await cc.takePicture();
      final Uint8List bytes = await photo.readAsBytes();
      try { File(photo.path).deleteSync(); } catch (_) {}
      if (_disposed) return;

      final req = http.MultipartRequest(
          'POST', Uri.parse('${ApiService.baseUrl}/nav_danger'));
      req.fields['language'] = widget.language;
      req.files.add(http.MultipartFile.fromBytes(
          'frame', bytes, filename: 'frame.jpg'));

      final streamed = await req.send().timeout(const Duration(seconds: 10));
      final body     = await streamed.stream.bytesToString();
      if (_disposed) return;

      final data = jsonDecode(body) as Map<String, dynamic>;
      _handleDangerResponse(data);
    } catch (e) {
      if (!_disposed) print('[Nav] _scanDangerOnce: $e');
    } finally {
      _cameraSendBusy = false;
    }
  }

  void _handleDangerResponse(Map<String, dynamic> data) {
    final ready     = data['ready'] != false;
    final speak     = (data['speak']     ?? '') as String;
    final danger    = (data['danger']    ?? 'LOW') as String;
    final interrupt = (data['interrupt'] ?? false) as bool;
    final rawObjs   = (data['objects']   ?? []) as List;

    if (!ready) {
      _yoloReady = false;
      // Cancel polling so we stop wasting battery on empty calls
      _dangerTimer?.cancel();
      print('[Nav] YOLO not ready on backend — danger detection disabled. '
            'Place best.pt at backend/models/yolo/best.pt');
      _ss(() {});
      return;
    }

    final boxes = <_NavDangerBox>[];
    for (final o in rawObjs) {
      if (o['is_danger'] != true) continue;
      final xyxy = (o['bbox_xyxy'] as List?) ?? const [];
      if (xyxy.length != 4) continue;
      final imgW = ((o['img_w'] ?? 1) as num).toDouble();
      final imgH = ((o['img_h'] ?? 1) as num).toDouble();
      final x1 = (xyxy[0] as num).toDouble();
      final y1 = (xyxy[1] as num).toDouble();
      final x2 = (xyxy[2] as num).toDouble();
      final y2 = (xyxy[3] as num).toDouble();
      final cx = ((x1 + x2) * 0.5) / imgW;
      final cy = ((y1 + y2) * 0.5) / imgH;
      final w  = (x2 - x1) / imgW;
      final h  = (y2 - y1) / imgH;
      boxes.add(_NavDangerBox(
        label:     (o['label']     ?? 'object') as String,
        proximity: (o['proximity'] ?? 'medium') as String,
        direction: (o['direction'] ?? 'center') as String,
        cx: cx, cy: cy, w: w, h: h,
      ));
    }

    _ss(() {
      _dangerStatus = danger;
      _dangerBoxes  = boxes;
    });

    if (speak.isNotEmpty && speak != _lastDangerSpoken) {
      _lastDangerSpoken = speak;
      TtsService.speak(speak,
          lang: widget.language,
          interrupt: danger == 'HIGH' || interrupt);
    }

    if (danger == 'HIGH') _vibrate(strong: true);
  }

  // ═══════════════════════════════════════════════════════════
  // OSRM POLYLINE + NOMINATIM (free)
  // ═══════════════════════════════════════════════════════════

  Future<void> _fetchOsrmPolyline(double oLat, double oLng) async {
    try {
      double? dLat;
      double? dLng;

      final geoUri1 = Uri.parse(
        '$_nominatimUrl'
        '?q=${Uri.encodeQueryComponent(widget.destination)}'
        '&format=json&limit=1&countrycodes=tn',
      );
      final geo1 = await http.get(geoUri1, headers: {
        'User-Agent': 'BlindGlassesApp/1.0 (student project)'
      }).timeout(const Duration(seconds: 10));

      List geoData = jsonDecode(geo1.body) as List;

      if (geoData.isEmpty) {
        final geoUri2 = Uri.parse(
          '$_nominatimUrl'
          '?q=${Uri.encodeQueryComponent(widget.destination)}'
          '&format=json&limit=1',
        );
        final geo2 = await http.get(geoUri2, headers: {
          'User-Agent': 'BlindGlassesApp/1.0 (student project)'
        }).timeout(const Duration(seconds: 10));
        geoData = jsonDecode(geo2.body) as List;
      }

      if (geoData.isEmpty) {
        print('[Nav] Nominatim: no results for "${widget.destination}"');
        return;
      }

      dLat = double.parse(geoData[0]['lat'] as String);
      dLng = double.parse(geoData[0]['lon'] as String);

      _ss(() => _destLatLng = LatLng(dLat!, dLng!));

      final profile = widget.navMode == 'walking' ? 'foot' : 'car';
      final osrmUri = Uri.parse(
        '$_osrmUrl/$profile/$oLng,$oLat;$dLng,$dLat'
        '?steps=false&geometries=geojson&overview=full',
      );
      final osrmResp =
          await http.get(osrmUri).timeout(const Duration(seconds: 15));
      final osrmData = jsonDecode(osrmResp.body) as Map<String, dynamic>;

      if (osrmData['code'] != 'Ok') {
        print('[Nav] OSRM error: ${osrmData['code']}');
        return;
      }

      final rawCoords =
          osrmData['routes'][0]['geometry']['coordinates'] as List;
      final coords = rawCoords
          .map((c) => LatLng((c[1] as num).toDouble(),
                              (c[0] as num).toDouble()))
          .toList();

      _ss(() => _routePoints = coords);

      if (coords.isNotEmpty && _mapReady) {
        try {
          _mapController.fitCamera(CameraFit.bounds(
              bounds: LatLngBounds.fromPoints(coords),
              padding: const EdgeInsets.all(52)));
        } catch (_) {}
      }
    } catch (e) {
      print('[Nav] _fetchOsrmPolyline: $e');
    }
  }

  // ═══════════════════════════════════════════════════════════
  // HTTP — backend /navigate
  // ═══════════════════════════════════════════════════════════

  Future<Map<String, dynamic>?> _callNavigate({
    required double lat, required double lng, required String dest,
  }) async {
    try {
      final req = http.MultipartRequest(
          'POST', Uri.parse('${ApiService.baseUrl}/navigate'));
      req.fields['lat']         = lat.toString();
      req.fields['lng']         = lng.toString();
      req.fields['destination'] = dest;
      req.fields['nav_mode']    = widget.navMode;
      req.fields['language']    = widget.language;

      final streamed =
          await req.send().timeout(const Duration(seconds: 20));
      final body = await streamed.stream.bytesToString();
      if (streamed.statusCode != 200) {
        print('[Nav] backend ${streamed.statusCode}: $body');
        return null;
      }
      return jsonDecode(body) as Map<String, dynamic>;
    } catch (e) {
      print('[Nav] _callNavigate: $e');
      return null;
    }
  }

  Future<void> _vibrate({required bool strong}) async {
    try {
      if (await Vibration.hasVibrator() ?? false) {
        Vibration.vibrate(pattern: strong ? [0, 300, 100, 300] : [0, 200]);
      }
    } catch (_) {}
  }

  void _close() {
    _gpsTimer?.cancel();
    _dangerTimer?.cancel();
    Navigator.pop(context);
  }

  // ═══════════════════════════════════════════════════════════
  // BUILD
  // ═══════════════════════════════════════════════════════════

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      body: SafeArea(
        child: Column(
          children: [
            _buildStatusBar(),
            Expanded(child: Stack(children: [
              _buildMap(),
              if (_cameraReady && _yoloReady) _buildCameraTile(),
            ])),
            _buildInstructionPanel(),
            _buildButtons(),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  // ── Status bar ────────────────────────────────────────────

  Widget _buildStatusBar() {
    final IconData icon;
    final String   label;
    if (_arrived) {
      icon = Icons.check_circle_rounded; label = 'Arrived!';
    } else if (_offRoute) {
      icon = Icons.warning_amber_rounded; label = 'OFF ROUTE';
    } else if (_dangerStatus == 'HIGH') {
      icon = Icons.error_rounded; label = 'HAZARD AHEAD';
    } else {
      icon = Icons.navigation_rounded;
      label = widget.navMode == 'walking' ? '🚶 Walking' : '🚗 Driving';
    }

    return AnimatedContainer(
      duration: const Duration(milliseconds: 400),
      margin: const EdgeInsets.fromLTRB(12, 10, 12, 0),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: _barColor.withOpacity(0.12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _barColor.withOpacity(0.5)),
      ),
      child: Row(children: [
        Icon(icon, color: _barColor, size: 17),
        const SizedBox(width: 10),
        Expanded(child: Text(label,
          style: TextStyle(color: _barColor, fontWeight: FontWeight.w700,
            fontSize: 13, letterSpacing: 1.1))),
        Flexible(child: Text(widget.destination,
          style: TextStyle(color: Colors.white.withOpacity(0.35),
            fontSize: 11),
          overflow: TextOverflow.ellipsis)),
      ]),
    );
  }

  // ── Embedded camera danger tile (top-right) ───────────────

  Widget _buildCameraTile() {
    return Positioned(
      top: 12, right: 12,
      child: Container(
        width: 130, height: 170,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: _dangerStatus == 'HIGH' ? _red
                 : _dangerStatus == 'MEDIUM' ? _amber
                 : _cyan.withOpacity(0.7),
            width: 2),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.5),
              blurRadius: 12, spreadRadius: 1)],
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: Stack(fit: StackFit.expand, children: [
            if (_camCtrl != null && _camCtrl!.value.isInitialized)
              CameraPreview(_camCtrl!)
            else
              const ColoredBox(color: Colors.black),
            if (_dangerBoxes.isNotEmpty)
              LayoutBuilder(builder: (ctx, cs) => CustomPaint(
                size: Size(cs.maxWidth, cs.maxHeight),
                painter: _NavBoxPainter(_dangerBoxes,
                  _dangerStatus == 'HIGH' ? _red : _amber),
              )),
            Positioned(top: 4, left: 4, child: Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.55),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                Container(width: 6, height: 6, decoration: BoxDecoration(
                  color: _dangerStatus == 'HIGH' ? _red
                       : _dangerStatus == 'MEDIUM' ? _amber : _green,
                  shape: BoxShape.circle)),
                const SizedBox(width: 4),
                const Text('LIVE', style: TextStyle(
                  color: Colors.white, fontSize: 9,
                  fontWeight: FontWeight.w700, letterSpacing: 1)),
              ]),
            )),
          ]),
        ),
      ),
    );
  }

  // ── OpenStreetMap tile map (free) ─────────────────────────

  Widget _buildMap() {
    final center = _userLatLng ?? const LatLng(36.8065, 10.1815);
    return ClipRect(
      child: FlutterMap(
        mapController: _mapController,
        options: MapOptions(
          initialCenter: center,
          initialZoom: 15.0,
          onMapReady: () {
            _mapReady = true;
            if (_routePoints.isNotEmpty) {
              try {
                _mapController.fitCamera(CameraFit.bounds(
                  bounds: LatLngBounds.fromPoints(_routePoints),
                  padding: const EdgeInsets.all(52)));
              } catch (_) {}
            }
          },
          interactionOptions: const InteractionOptions(
            flags: InteractiveFlag.pinchZoom | InteractiveFlag.drag,
          ),
        ),
        children: [
          TileLayer(
            urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            userAgentPackageName: 'com.example.voice_assistant',
          ),
          if (_routePoints.isNotEmpty)
            PolylineLayer(polylines: [Polyline(
              points: _routePoints,
              color: _cyan.withOpacity(0.85),
              strokeWidth: 5.0,
              borderColor: Colors.white.withOpacity(0.25),
              borderStrokeWidth: 1.5,
            )]),
          if (_destLatLng != null)
            MarkerLayer(markers: [Marker(
              point: _destLatLng!, width: 40, height: 40,
              child: const Icon(Icons.place_rounded,
                  color: _red, size: 40),
            )]),
          if (_userLatLng != null)
            MarkerLayer(markers: [Marker(
              point: _userLatLng!, width: 24, height: 24,
              child: Container(decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _cyan,
                border: Border.all(color: Colors.white, width: 2.5),
                boxShadow: [BoxShadow(
                  color: _cyan.withOpacity(0.5),
                  blurRadius: 10, spreadRadius: 2)],
              )),
            )]),
          const RichAttributionWidget(attributions: [
            TextSourceAttribution('OpenStreetMap contributors'),
          ]),
        ],
      ),
    );
  }

  // ── Instruction panel ─────────────────────────────────────

  Widget _buildInstructionPanel() {
    return Container(
      color: _bg,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Row(crossAxisAlignment: CrossAxisAlignment.center, children: [
          Icon(_navIcon(_statusText), color: _cyan, size: 36),
          const SizedBox(width: 14),
          Expanded(child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 300),
            child: Text(_statusText, key: ValueKey(_statusText),
              style: const TextStyle(
                color: Colors.white, fontSize: 16,
                fontWeight: FontWeight.w600, height: 1.4)),
          )),
        ]),
        if (_etaText.isNotEmpty) ...[
          const SizedBox(height: 6),
          Align(alignment: Alignment.centerLeft, child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: _cyan.withOpacity(0.1),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: _cyan.withOpacity(0.3))),
            child: Text(_etaText, style: TextStyle(
              color: Colors.white.withOpacity(0.6), fontSize: 12)),
          )),
        ],
        if (_started) ...[
          const SizedBox(height: 4),
          Align(alignment: Alignment.centerLeft, child: Text(
            'Step ${_currentStep + 1}',
            style: TextStyle(color: Colors.white.withOpacity(0.25),
                fontSize: 11))),
        ],
      ]),
    );
  }

  // ── Buttons ───────────────────────────────────────────────

  Widget _buildButtons() => Padding(
    padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
    child: Row(children: [
      Expanded(child: _ActionButton(
        icon: Icons.refresh_rounded, label: 'Recalculate', color: _cyan,
        onTap: () async {
          final pos = await LocationService.current();
          if (pos == null) return;
          _ss(() => _statusText = 'Recalculating…');
          await _fetchOsrmPolyline(pos.latitude, pos.longitude);
          final result = await _callNavigate(
              lat: pos.latitude, lng: pos.longitude,
              dest: widget.destination);
          if (result != null) {
            final speak = (result['speak'] ?? '') as String;
            _ss(() => _statusText = speak);
            if (speak.isNotEmpty && speak != _lastSpoken) {
              _lastSpoken = speak;
              TtsService.speak(speak, lang: widget.language);
            }
          }
        },
      )),
      const SizedBox(width: 12),
      Expanded(child: _ActionButton(
        icon: Icons.close_rounded, label: 'Stop',
        color: Colors.white.withOpacity(0.3),
        onTap: _close,
      )),
    ]),
  );

  IconData _navIcon(String text) {
    final t = text.toLowerCase();
    if (t.contains('left'))    return Icons.turn_left_rounded;
    if (t.contains('right'))   return Icons.turn_right_rounded;
    if (t.contains('arrived')) return Icons.place_rounded;
    if (t.contains('warning') || t.contains('off'))
      return Icons.warning_amber_rounded;
    if (t.contains('u-turn')) return Icons.u_turn_left_rounded;
    return Icons.straight_rounded;
  }
}

// ─── Reusable action button ───────────────────────────────────

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;
  const _ActionButton({
    required this.icon, required this.label,
    required this.color, required this.onTap,
  });
  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: onTap,
    child: Container(
      padding: const EdgeInsets.symmetric(vertical: 13),
      decoration: BoxDecoration(
        color: color.withOpacity(0.10),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.35))),
      child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
        Icon(icon, color: color, size: 17),
        const SizedBox(width: 8),
        Text(label, style: TextStyle(
          color: color, fontSize: 14, fontWeight: FontWeight.w600)),
      ]),
    ),
  );
}

// ─── Tiny bbox painter for the camera tile ────────────────────

class _NavBoxPainter extends CustomPainter {
  final List<_NavDangerBox> boxes;
  final Color color;
  const _NavBoxPainter(this.boxes, this.color);

  @override
  void paint(Canvas canvas, Size size) {
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = color;
    final fill = Paint()
      ..style = PaintingStyle.fill
      ..color = color.withOpacity(0.15);

    for (final b in boxes) {
      final cx = b.cx * size.width;
      final cy = b.cy * size.height;
      final w  = b.w  * size.width;
      final h  = b.h  * size.height;
      final rect = Rect.fromCenter(center: Offset(cx, cy),
          width: w, height: h);
      final rr = RRect.fromRectAndRadius(rect, const Radius.circular(4));
      canvas.drawRRect(rr, fill);
      canvas.drawRRect(rr, stroke);

      // Tiny label on top
      final tp = TextPainter(
        text: TextSpan(text: b.label,
          style: const TextStyle(color: Colors.white,
            fontSize: 8, fontWeight: FontWeight.w700,
            shadows: [Shadow(blurRadius: 2, color: Colors.black)])),
        textDirection: TextDirection.ltr)..layout();
      tp.paint(canvas, Offset(rect.left + 2, rect.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant _NavBoxPainter old) =>
      old.boxes != boxes || old.color != color;
}