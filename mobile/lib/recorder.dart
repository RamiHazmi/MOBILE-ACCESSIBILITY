import 'dart:async';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'tts_service.dart';
import 'alarm_service.dart';

class RecorderService {
  static final AudioRecorder _recorder = AudioRecorder();
  static bool    _isRecording = false;
  static String? _filePath;

  // ── Amplitude stream ────────────────────────────────────────
  // Emits normalised amplitude values [0.0 – 1.0] while recording.
  // ui_home.dart subscribes to this to drive the waveform painter.
  static final StreamController<double> _ampController =
      StreamController<double>.broadcast();

  static Stream<double> get amplitudeStream => _ampController.stream;

  static Timer? _ampTimer;

  static void _startAmplitudePolling() {
    _ampTimer?.cancel();
    _ampTimer = Timer.periodic(const Duration(milliseconds: 80), (_) async {
      if (!_isRecording) return;
      try {
        final amp = await _recorder.getAmplitude();
        // amp.current is in dBFS (negative). Map [-60, 0] → [0.0, 1.0].
        final db        = amp.current.clamp(-60.0, 0.0);
        final normalised = (db + 60.0) / 60.0;
        _ampController.add(normalised);
      } catch (_) {}
    });
  }

  static void _stopAmplitudePolling() {
    _ampTimer?.cancel();
    _ampTimer = null;
    _ampController.add(0.0); // reset waveform to flat
  }

  static Future<String?> startRecording() async {
    print('🎤 startRecording()');
    if (_isRecording) return _filePath;

    // Stop any ongoing TTS immediately so the mic picks up the user, not the agent.
    await TtsService.stop();

    // Pause alarm monitoring so we get exclusive mic access.
    await AlarmService.instance.pause();

    if (!await _recorder.hasPermission()) {
      print('❌ Mic permission denied');
      return null;
    }

    final dir = await getTemporaryDirectory();
    _filePath = '${dir.path}/audio_record.pcm';

    final file = File(_filePath!);
    if (file.existsSync()) await file.delete();

    await _recorder.start(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
      ),
      path: _filePath!,
    );
    _isRecording = true;
    _startAmplitudePolling();
    print('🔴 Recording started → $_filePath');
    return _filePath;
  }

  static Future<String?> stopRecording() async {
    if (!_isRecording) return _filePath;
    _stopAmplitudePolling();
    await _recorder.stop();
    _isRecording = false;
    if (_filePath != null) {
      final f = File(_filePath!);
      if (f.existsSync()) {
        print('✅ Saved → $_filePath (${f.lengthSync()} bytes)');
      } else {
        _filePath = null;
      }
    }
    // Resume alarm monitoring now that we've released the mic.
    AlarmService.instance.resume();
    return _filePath;
  }

  static bool get isRecording => _isRecording;
}