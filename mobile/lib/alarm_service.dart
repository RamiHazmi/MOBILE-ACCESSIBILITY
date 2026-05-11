// mobile/lib/alarm_service.dart
// ════════════════════════════════════════════════════════════════
// Singleton that records 1.5s audio chunks and streams them to the
// backend YAMNet alarm detector via WebSocket.
//
// Uses a poll-loop (start→stop→send→repeat) instead of startStream,
// which is unreliable on many Android devices.
//
// Mic coordination with main recorder:
//   RecorderService.startRecording() → AlarmService.pause()
//   RecorderService.stopRecording()  → AlarmService.resume()
// ════════════════════════════════════════════════════════════════

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:vibration/vibration.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as ws_status;

import 'api_service.dart';

// ── Data model ───────────────────────────────────────────────────
class AlarmEvent {
  final String   label;
  final String   emoji;
  final String   category;
  final String   description;
  final double   confidence;
  final String   ts;
  final DateTime receivedAt;

  AlarmEvent({
    required this.label,
    required this.emoji,
    required this.category,
    required this.description,
    required this.confidence,
    required this.ts,
  }) : receivedAt = DateTime.now();
}

// ── Singleton ────────────────────────────────────────────────────
class AlarmService {
  AlarmService._();
  static final AlarmService instance = AlarmService._();

  /// Latest unread alert — null when none.  Subscribe via addListener.
  final ValueNotifier<AlarmEvent?>       latestAlert = ValueNotifier(null);
  final ValueNotifier<List<AlarmEvent>>  history     = ValueNotifier([]);

  String get _wsUrl =>
      ApiService.baseUrl.replaceFirst(RegExp(r'^http'), 'ws') + '/ws/alarm';

  WebSocketChannel?   _channel;
  StreamSubscription? _wsSub;
  final AudioRecorder _recorder = AudioRecorder();

  bool _running     = false;
  bool _paused      = false;
  bool _wsReady     = false;
  bool _polling     = false;
  bool _connecting  = false;  // guard against concurrent connect attempts

  // ── Public API ──────────────────────────────────────────────

  Future<void> start() async {
    if (_running) return;
    _running = true;
    print('[alarm_svc] starting');
    await _connectWs();
  }

  /// Called by RecorderService before it takes the mic.
  Future<void> pause() async {
    if (!_running || _paused) return;
    _paused  = true;
    _polling = false;
    _channel?.sink.add('STOP');
    try { await _recorder.stop(); } catch (_) {}
    print('[alarm_svc] paused (mic handed to recorder)');
  }

  /// Called by RecorderService after it releases the mic.
  Future<void> resume() async {
    if (!_running || !_paused) return;
    _paused = false;
    print('[alarm_svc] resuming');
    if (_wsReady) await _startAudio();
  }

  void dismissAlert() => latestAlert.value = null;

  void stop() {
    _running    = false;
    _paused     = false;
    _polling    = false;
    _connecting = false;
    try { _recorder.stop(); } catch (_) {}
    _wsSub?.cancel();
    _wsSub = null;
    try { _channel?.sink.close(ws_status.goingAway); } catch (_) {}
    _channel = null;
    print('[alarm_svc] stopped');
  }

  // ── WebSocket ────────────────────────────────────────────────

  Future<void> _connectWs() async {
    if (_connecting) return;  // already trying to connect
    _connecting = true;
    // Close any existing connection first
    _wsSub?.cancel();
    _wsSub = null;
    try { _channel?.sink.close(ws_status.goingAway); } catch (_) {}
    _channel = null;
    try {
      _channel = WebSocketChannel.connect(Uri.parse(_wsUrl));
      _wsSub   = _channel!.stream.listen(
        _onWsMessage,
        onError: (e) {
          print('[alarm_svc] WS error: $e');
          _connecting = false;
          _scheduleReconnect();
        },
        onDone: () {
          print('[alarm_svc] WS closed');
          _connecting = false;
          _scheduleReconnect();
        },
      );
      _connecting = false;
      print('[alarm_svc] WS connected to $_wsUrl');
    } catch (e) {
      _connecting = false;
      print('[alarm_svc] WS connect failed: $e');
      _scheduleReconnect();
    }
  }

  void _scheduleReconnect() {
    _wsReady = false;
    _polling = false;
    if (!_running) return;
    print('[alarm_svc] reconnecting in 6s…');
    Future.delayed(const Duration(seconds: 6), () {
      if (_running && !_paused && !_connecting) _connectWs();
    });
  }

  void _onWsMessage(dynamic raw) {
    if (raw is! String) return;
    final Map<String, dynamic> msg;
    try { msg = jsonDecode(raw) as Map<String, dynamic>; } catch (_) { return; }

    final type = msg['type'] as String? ?? '';
    print('[alarm_svc] ← $type  ${msg.toString().substring(0, msg.toString().length.clamp(0, 120))}');

    switch (type) {
      case 'ready':
        final ok = msg['status'] == 'ok';
        _wsReady = ok;
        if (ok) {
          print('[alarm_svc] server ready, starting audio');
          if (!_paused) _startAudio();
        } else {
          // Model not yet loaded on server — retry in 8s
          print('[alarm_svc] server not ready yet, retrying in 8s');
          Future.delayed(const Duration(seconds: 8), () {
            if (_running && !_paused) _scheduleReconnect();
          });
        }
        break;

      case 'detection':
        final event = AlarmEvent(
          label:       msg['label']       as String? ?? '',
          emoji:       msg['emoji']       as String? ?? '⚠️',
          category:    msg['category']    as String? ?? 'danger',
          description: msg['description'] as String? ?? '',
          confidence:  (msg['confidence'] as num?)?.toDouble() ?? 0,
          ts:          msg['ts']          as String? ?? '',
        );
        print('[alarm_svc] 🔔 DETECTION: ${event.label} (${(event.confidence * 100).toStringAsFixed(0)}%)');
        _onDetection(event);
        break;

      case 'status':
        print('[alarm_svc] status: ${msg['message']}');
        break;

      case 'error':
        print('[alarm_svc] server error: ${msg['message']}');
        break;
    }
  }

  void _onDetection(AlarmEvent event) {
    Vibration.vibrate(pattern: [0, 300, 120, 300, 120, 600]);
    final updated = [event, ...history.value];
    if (updated.length > 100) updated.removeLast();
    history.value     = updated;
    latestAlert.value = event;
  }

  // ── Audio polling loop ────────────────────────────────────────
  // Records 1.5s → sends bytes → repeats.
  // This is more reliable than startStream() on Android.

  Future<void> _startAudio() async {
    if (_paused || !_running || _polling) return;
    if (!await _recorder.hasPermission()) {
      print('[alarm_svc] mic permission denied');
      return;
    }
    _polling = true;
    _channel?.sink.add('START');
    print('[alarm_svc] audio poll loop starting');
    _pollLoop();
  }

  void _pollLoop() async {
    String? tmpPath;
    try {
      final dir = await getTemporaryDirectory();
      tmpPath = '${dir.path}/alarm_chunk.pcm';
    } catch (e) {
      print('[alarm_svc] temp dir error: $e');
      _polling = false;
      return;
    }

    while (_running && !_paused && _polling) {
      try {
        // Delete any leftover file
        final f = File(tmpPath);
        if (f.existsSync()) f.deleteSync();

        // Record 1.5 seconds
        await _recorder.start(
          const RecordConfig(
            encoder:     AudioEncoder.pcm16bits,
            sampleRate:  16000,
            numChannels: 1,
          ),
          path: tmpPath,
        );

        await Future.delayed(const Duration(milliseconds: 1500));

        if (!_polling || _paused) {
          try { await _recorder.stop(); } catch (_) {}
          break;
        }

        await _recorder.stop();

        if (!_polling || _paused || !_running) break;

        // Read and send the chunk
        final file = File(tmpPath);
        if (file.existsSync()) {
          final bytes = file.readAsBytesSync();
          print('[alarm_svc] → sending ${bytes.length} bytes');
          if (bytes.length > 3200 && _channel != null) {
            try {
              _channel!.sink.add(Uint8List.fromList(bytes));
            } catch (e) {
              print('[alarm_svc] send error: $e');
            }
          }
          try { file.deleteSync(); } catch (_) {}
        } else {
          print('[alarm_svc] chunk file missing after stop');
        }

        // Small gap between chunks
        await Future.delayed(const Duration(milliseconds: 100));

      } catch (e) {
        print('[alarm_svc] poll loop error: $e');
        await Future.delayed(const Duration(milliseconds: 800));
      }
    }

    print('[alarm_svc] poll loop ended (polling=$_polling paused=$_paused running=$_running)');
  }
}
