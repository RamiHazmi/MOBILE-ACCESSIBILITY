// save_object_screen.dart
//
// Flow:
//   1. Camera opens — TTS: "Point at [name] and tap to capture."
//   2. User taps → single frame captured → sent to /describe_for_save
//   3. TTS reads VLM description + "Tap to save, or tap Retake."
//   4. Tap screen → save → TTS "Saved!" → close
//      Tap Retake  → back to step 1

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'api_service.dart';
import 'tts_service.dart';

enum _SaveState { idle, capturing, describing, confirming, saving, done }

class SaveObjectScreen extends StatefulWidget {
  final String objectName;
  final String language;
  const SaveObjectScreen({
    super.key,
    required this.objectName,
    required this.language,
  });
  @override
  State<SaveObjectScreen> createState() => _SaveObjectScreenState();
}

class _SaveObjectScreenState extends State<SaveObjectScreen>
    with SingleTickerProviderStateMixin {
  CameraController? _ctrl;
  bool _camReady  = false;
  bool _disposed  = false;

  _SaveState _state       = _SaveState.idle;
  String     _statusText  = '';
  String     _description = '';

  late AnimationController _pulseAnim;

  static const Color _accent = Color(0xFF00E5C0);

  @override
  void initState() {
    super.initState();
    _pulseAnim = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1200))
      ..repeat(reverse: true);
    _initCamera();
  }

  @override
  void dispose() {
    _disposed = true;
    _pulseAnim.dispose();
    final ctrl = _ctrl; _ctrl = null;
    Future.delayed(const Duration(milliseconds: 300),
        () async { try { await ctrl?.dispose(); } catch (_) {} });
    super.dispose();
  }

  void _ss(VoidCallback fn) {
    if (!_disposed && mounted) setState(fn);
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty || _disposed) return;
      final ctrl = CameraController(cameras.first, ResolutionPreset.medium,
          enableAudio: false);
      await ctrl.initialize();
      await ctrl.setFlashMode(FlashMode.off);
      if (_disposed) { await ctrl.dispose(); return; }
      _ctrl = ctrl;
      _ss(() { _camReady = true; _statusText = ''; });
      await TtsService.speak(
          'Point the camera at ${widget.objectName} and tap to capture.',
          lang: widget.language);
      _ss(() => _statusText =
          'Point at ${widget.objectName} — tap to capture');
    } catch (e) {
      _ss(() => _statusText = 'Camera unavailable');
    }
  }

  Future<void> _onTap() async {
    if (_state == _SaveState.idle && _camReady) {
      await _capture();
    } else if (_state == _SaveState.confirming) {
      await _saveObject();
    }
  }

  Future<void> _capture() async {
    _ss(() { _state = _SaveState.capturing; _statusText = 'Capturing…'; });
    HapticFeedback.lightImpact();

    Uint8List? bytes;
    try {
      final xfile = await _ctrl!.takePicture();
      bytes = await xfile.readAsBytes();
    } catch (e) {
      _ss(() { _state = _SaveState.idle; _statusText = 'Capture failed. Tap to retry.'; });
      return;
    }

    _ss(() { _state = _SaveState.describing; _statusText = 'Analyzing…'; });

    final raw  = await ApiService.describeForSave(
        frameBytes: bytes,
        objectName: widget.objectName,
        language:   widget.language);
    final data = jsonDecode(raw) as Map<String, dynamic>;

    if (data.containsKey('error')) {
      _ss(() { _state = _SaveState.idle; _statusText = 'Could not analyze. Tap to retry.'; });
      await TtsService.speak('Could not analyze the image. Tap to retry.',
          lang: widget.language);
      return;
    }

    final desc  = (data['description'] as String? ?? '').trim();
    final speak = (data['speak']       as String? ?? '').trim();

    _ss(() {
      _description = desc;
      _state       = _SaveState.confirming;
      _statusText  = speak.isNotEmpty ? speak : desc;
    });

    if (speak.isNotEmpty) {
      await TtsService.speak(speak, lang: widget.language);
    }
  }

  Future<void> _retake() async {
    _ss(() {
      _state       = _SaveState.idle;
      _description = '';
      _statusText  = 'Point at ${widget.objectName} — tap to capture';
    });
    await TtsService.speak(
        'Point the camera at ${widget.objectName} and tap to capture.',
        lang: widget.language);
  }

  Future<void> _saveObject() async {
    _ss(() { _state = _SaveState.saving; _statusText = 'Saving…'; });
    HapticFeedback.mediumImpact();

    final raw  = await ApiService.savePersonalObject(
        name:        widget.objectName,
        description: _description,
        language:    widget.language);
    final data = jsonDecode(raw) as Map<String, dynamic>;
    final speak = (data['speak'] as String? ?? 'Saved!').trim();

    _ss(() { _state = _SaveState.done; _statusText = speak; });
    await TtsService.speak(speak, lang: widget.language);
    await Future.delayed(const Duration(seconds: 2));
    if (mounted) Navigator.pop(context);
  }

  // ── Build ─────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final camOk = _camReady && _ctrl != null
        && !_disposed && _ctrl!.value.isInitialized;

    return Scaffold(
      backgroundColor: Colors.black,
      body: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: (_state == _SaveState.idle || _state == _SaveState.confirming)
            ? _onTap
            : null,
        onDoubleTap: _state == _SaveState.confirming ? _retake : null,
        child: Stack(fit: StackFit.expand, children: [
          // Camera or loading
          camOk ? CameraPreview(_ctrl!) : _buildLoading(),

          // Dark vignette
          DecoratedBox(
            decoration: BoxDecoration(
              gradient: RadialGradient(
                center: Alignment.center, radius: 1.0,
                colors: [Colors.transparent, Colors.black.withOpacity(0.4)]),
            ),
          ),

          // Top bar
          _buildTopBar(),

          // Scan ring when idle
          if (_state == _SaveState.idle && camOk) _buildScanRing(),

          // Bottom panel — status + buttons
          _buildBottomPanel(),
        ]),
      ),
    );
  }

  Widget _buildLoading() => Container(
    color: const Color(0xFF080810),
    child: const Center(child: CircularProgressIndicator(
        color: _accent, strokeWidth: 2)),
  );

  Widget _buildTopBar() => Positioned(
    top: 0, left: 0, right: 0,
    child: SafeArea(
      bottom: false,
      child: Container(
        margin:  const EdgeInsets.fromLTRB(12, 8, 12, 0),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: _accent.withOpacity(0.12),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: _accent.withOpacity(0.5), width: 1),
        ),
        child: Row(children: [
          const Icon(Icons.bookmark_add_rounded, color: _accent, size: 16),
          const SizedBox(width: 8),
          Text('SAVE  ·  ${widget.objectName.toUpperCase()}',
              style: const TextStyle(
                  color: _accent, fontSize: 12,
                  fontWeight: FontWeight.w700, letterSpacing: 1.2)),
          const Spacer(),
          GestureDetector(
            onTap: () => Navigator.pop(context),
            child: const Icon(Icons.close_rounded,
                color: Colors.white54, size: 20),
          ),
        ]),
      ),
    ),
  );

  Widget _buildScanRing() => Center(
    child: AnimatedBuilder(
      animation: _pulseAnim,
      builder: (_, __) => Transform.scale(
        scale: 1.0 + _pulseAnim.value * 0.06,
        child: Container(
          width: 220, height: 220,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(
              color: _accent.withOpacity(0.2 + _pulseAnim.value * 0.2),
              width: 1.5)),
        ),
      ),
    ),
  );

  Widget _buildBottomPanel() => Positioned(
    bottom: 0, left: 0, right: 0,
    child: Container(
      padding: const EdgeInsets.fromLTRB(20, 28, 20, 44),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter, end: Alignment.topCenter,
          colors: [Colors.black.withOpacity(0.95),
                   Colors.black.withOpacity(0.6), Colors.transparent],
          stops: const [0.0, 0.6, 1.0])),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        if (_statusText.isNotEmpty)
          Text(_statusText,
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: Colors.white, fontSize: 16,
              fontWeight: FontWeight.w500, height: 1.45,
              shadows: [Shadow(blurRadius: 8, color: Colors.black)])),

        const SizedBox(height: 20),

        if (_state == _SaveState.confirming) ...[
          const Text(
            '• tap once to save  •• double tap to retake',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white38,
              fontSize: 13,
              height: 1.5,
              letterSpacing: 0.2,
            ),
          ),
        ] else if (_state == _SaveState.idle || _state == _SaveState.done) ...[
          _BigButton(
            label: _state == _SaveState.done ? 'Done' : 'Tap to capture',
            icon:  _state == _SaveState.done
                ? Icons.check_rounded
                : Icons.camera_alt_rounded,
            color: _state == _SaveState.done ? _accent : Colors.white38,
            onTap: _state == _SaveState.done
                ? () => Navigator.pop(context)
                : null,
          ),
        ] else ...[
          const SizedBox(
            width: 28, height: 28,
            child: CircularProgressIndicator(
                color: _accent, strokeWidth: 2)),
        ],
      ]),
    ),
  );
}

class _BigButton extends StatelessWidget {
  final String    label;
  final IconData  icon;
  final Color     color;
  final VoidCallback? onTap;
  const _BigButton({
    required this.label, required this.icon,
    required this.color, this.onTap});

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: onTap,
    child: Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 14),
      decoration: BoxDecoration(
        color:        color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(14),
        border:       Border.all(color: color.withOpacity(0.45), width: 1),
      ),
      child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(width: 10),
        Text(label,
            style: TextStyle(
                color: color, fontSize: 15, fontWeight: FontWeight.w600)),
      ]),
    ),
  );
}
