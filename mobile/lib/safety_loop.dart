import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'camera_service.dart';
import 'tts_service.dart';

typedef DangerCallback = void Function(
    String danger, String speak, bool interrupt);

class SafetyLoop {
  static const String   _baseUrl  = 'http://192.168.1.13:8000';
  static const Duration _interval = Duration(seconds: 3);

  Timer? _timer;
  final DangerCallback onDanger;

  bool _running = false;
  bool _busy    = false;   // prevents re-entrant ticks

  SafetyLoop({required this.onDanger});

  // ── Control ───────────────────────────────────────────────────

  void start() {
    if (_running) return;
    _running = true;
    print('🛡 Safety loop started');
    _timer = Timer.periodic(_interval, (_) => _tick());
  }

  void stop() {
    _timer?.cancel();
    _timer   = null;
    _running = false;
    _busy    = false;
    print('🛡 Safety loop stopped');
  }

  // ── Tick ──────────────────────────────────────────────────────

  Future<void> _tick() async {
    // Skip if previous tick is still running or loop was stopped
    if (_busy || !_running) return;
    _busy = true;

    try {
      await _doTick();
    } catch (e) {
      print('🛡 Safety tick error: $e');
    } finally {
      _busy = false;
    }
  }

  Future<void> _doTick() async {
    // CameraService.captureFrame() already has its own mutex
    // and returns null immediately when not ready or busy.
    if (!CameraService.isReady) return;

    final Uint8List? frameBytes = await CameraService.captureFrame();
    if (frameBytes == null || frameBytes.isEmpty) return;

    final uri     = Uri.parse('$_baseUrl/safety');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(http.MultipartFile.fromBytes(
      'frame', frameBytes,
      filename: 'frame.jpg',
    ));

    final streamed = await request
        .send()
        .timeout(const Duration(seconds: 14));
    final body = await streamed.stream.bytesToString();
    if (streamed.statusCode != 200) return;

    final data      = jsonDecode(body) as Map<String, dynamic>;
    final danger    = (data['danger']    ?? 'LOW')  as String;
    final speak     = (data['speak']     ?? '')     as String;
    final interrupt = (data['interrupt'] ?? false)  as bool;

    // Notify UI
    onDanger(danger, speak, interrupt);

    // Only speak non-LOW alerts
    if (speak.isNotEmpty && danger != 'LOW') {
      await TtsService.speak(speak, lang: 'en', interrupt: interrupt);
    }
  }
}