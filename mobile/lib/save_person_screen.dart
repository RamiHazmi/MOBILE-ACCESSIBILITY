import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'api_service.dart';
import 'tts_service.dart';

enum _PersonState { idle, capturing, describing, confirming, saving, done }

class SavePersonScreen extends StatefulWidget {
  final String personName;
  final String relationship;
  final String language;
  const SavePersonScreen({
    super.key,
    required this.personName,
    required this.relationship,
    required this.language,
  });
  @override
  State<SavePersonScreen> createState() => _SavePersonScreenState();
}

class _SavePersonScreenState extends State<SavePersonScreen>
    with SingleTickerProviderStateMixin {
  CameraController? _ctrl;
  bool _camReady  = false;
  bool _disposed  = false;

  _PersonState _state         = _PersonState.idle;
  String       _statusText   = '';
  String       _description  = '';
  Uint8List?   _capturedBytes;  // raw JPEG from camera — sent with save

  late AnimationController _pulseAnim;

  static const Color _accent = Color(0xFFFF8A65);  // warm orange for people

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
      final isSelf  = widget.relationship.toLowerCase() == 'self';
      // Front camera for selfie (saving own face), rear for saving others
      final target  = isSelf
          ? CameraLensDirection.front
          : CameraLensDirection.back;
      final front = cameras.firstWhere(
        (c) => c.lensDirection == target,
        orElse: () => cameras.first,
      );
      if (_disposed) return;
      final ctrl = CameraController(front, ResolutionPreset.medium,
          enableAudio: false);
      await ctrl.initialize();
      await ctrl.setFlashMode(FlashMode.off);
      if (_disposed) { await ctrl.dispose(); return; }
      _ctrl = ctrl;
      _ss(() { _camReady = true; _statusText = ''; });
      final _isSelf = widget.relationship.toLowerCase() == 'self';
      await TtsService.speak(
          _isSelf
              ? 'Point the front camera at your face and tap to capture.'
              : 'Point the camera at ${widget.personName}\'s face and tap to capture.',
          lang: widget.language);
      _ss(() => _statusText = _isSelf
          ? 'Point the camera at your face — tap to capture'
          : 'Point at face — tap to capture');
    } catch (e) {
      _ss(() => _statusText = 'Camera unavailable');
    }
  }

  Future<void> _onTap() async {
    if (_state == _PersonState.idle && _camReady) {
      await _capture();
    } else if (_state == _PersonState.confirming) {
      await _savePerson();
    }
  }

  Future<void> _capture() async {
    _ss(() { _state = _PersonState.capturing; _statusText = 'Capturing…'; });
    HapticFeedback.lightImpact();

    Uint8List? bytes;
    try {
      final xfile = await _ctrl!.takePicture();
      bytes = await xfile.readAsBytes();
    } catch (e) {
      _ss(() { _state = _PersonState.idle; _statusText = 'Capture failed. Tap to retry.'; });
      return;
    }

    _capturedBytes = bytes;  // keep for saving
    _ss(() { _state = _PersonState.describing; _statusText = 'Analyzing face…'; });

    final raw  = await ApiService.describeForSavePerson(
        frameBytes:   bytes,
        personName:   widget.personName,
        relationship: widget.relationship,
        language:     widget.language);
    final data = jsonDecode(raw) as Map<String, dynamic>;

    if (data.containsKey('error')) {
      _ss(() { _state = _PersonState.idle; _statusText = 'Could not analyze. Tap to retry.'; });
      await TtsService.speak('Could not analyze the image. Tap to retry.',
          lang: widget.language);
      return;
    }

    final desc  = (data['description'] as String? ?? '').trim();
    final speak = (data['speak']       as String? ?? '').trim();

    _ss(() {
      _description = desc;
      _state       = _PersonState.confirming;
      _statusText  = speak.isNotEmpty ? speak : desc;
    });

    if (speak.isNotEmpty) {
      await TtsService.speak(speak, lang: widget.language);
    }
  }

  Future<void> _retake() async {
    final _isSelf = widget.relationship.toLowerCase() == 'self';
    _capturedBytes = null;
    _ss(() {
      _state       = _PersonState.idle;
      _description = '';
      _statusText  = _isSelf
          ? 'Point the camera at your face — tap to capture'
          : 'Point at face — tap to capture';
    });
    await TtsService.speak(
        _isSelf
            ? 'Point the front camera at your face and tap to capture.'
            : 'Point the camera at ${widget.personName}\'s face and tap to capture.',
        lang: widget.language);
  }

  Future<void> _savePerson() async {
    _ss(() { _state = _PersonState.saving; _statusText = 'Saving…'; });
    HapticFeedback.mediumImpact();

    final imageB64 = _capturedBytes != null
        ? base64Encode(_capturedBytes!)
        : '';

    final raw  = await ApiService.savePersonRecord(
        name:         widget.personName,
        relationship: widget.relationship,
        description:  _description,
        imageB64:     imageB64,
        language:     widget.language);
    final data = jsonDecode(raw) as Map<String, dynamic>;
    final speak = (data['speak'] as String? ?? 'Saved!').trim();

    _ss(() { _state = _PersonState.done; _statusText = speak; });
    await TtsService.speak(speak, lang: widget.language);
    await Future.delayed(const Duration(seconds: 2));
    if (mounted) Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    final camOk = _camReady && _ctrl != null
        && !_disposed && _ctrl!.value.isInitialized;

    return Scaffold(
      backgroundColor: Colors.black,
      body: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: (_state == _PersonState.idle || _state == _PersonState.confirming)
            ? _onTap
            : null,
        onDoubleTap: _state == _PersonState.confirming ? _retake : null,
        child: Stack(fit: StackFit.expand, children: [
          camOk ? CameraPreview(_ctrl!) : _buildLoading(),
          DecoratedBox(
            decoration: BoxDecoration(
              gradient: RadialGradient(
                center: Alignment.center, radius: 1.0,
                colors: [Colors.transparent, Colors.black.withOpacity(0.4)]),
            ),
          ),
          _buildTopBar(),
          if (_state == _PersonState.idle && camOk) _buildFaceGuide(),
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
          const Icon(Icons.person_add_rounded, color: _accent, size: 16),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              widget.relationship.toLowerCase() == 'self'
                  ? 'SAVE  ·  YOUR FACE'
                  : 'SAVE  ·  ${widget.personName.toUpperCase()}'
                    '  ·  ${widget.relationship.toUpperCase()}',
              style: const TextStyle(
                  color: _accent, fontSize: 12,
                  fontWeight: FontWeight.w700, letterSpacing: 1.2),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          const SizedBox(width: 8),
          GestureDetector(
            onTap: () => Navigator.pop(context),
            child: const Icon(Icons.close_rounded,
                color: Colors.white54, size: 20),
          ),
        ]),
      ),
    ),
  );

  // Oval face guide
  Widget _buildFaceGuide() => Center(
    child: AnimatedBuilder(
      animation: _pulseAnim,
      builder: (_, __) => Transform.scale(
        scale: 1.0 + _pulseAnim.value * 0.04,
        child: Container(
          width: 200, height: 260,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(120),
            border: Border.all(
              color: _accent.withOpacity(0.25 + _pulseAnim.value * 0.2),
              width: 1.5),
          ),
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

        if (_state == _PersonState.confirming) ...[
          const Text(
            '• tap once to save  •• double tap to retake',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white38, fontSize: 13,
              height: 1.5, letterSpacing: 0.2),
          ),
        ] else if (_state == _PersonState.idle || _state == _PersonState.done) ...[
          _PersonBtn(
            label: _state == _PersonState.done
                ? 'Done'
                : 'Tap to capture',
            icon:  _state == _PersonState.done
                ? Icons.check_rounded
                : Icons.camera_alt_rounded,
            color: _state == _PersonState.done ? _accent : Colors.white38,
            onTap: _state == _PersonState.done
                ? () => Navigator.pop(context)
                : null,
          ),
        ] else ...[
          const SizedBox(
            width: 28, height: 28,
            child: CircularProgressIndicator(color: _accent, strokeWidth: 2)),
        ],
      ]),
    ),
  );
}

class _PersonBtn extends StatelessWidget {
  final String     label;
  final IconData   icon;
  final Color      color;
  final VoidCallback? onTap;
  const _PersonBtn({required this.label, required this.icon,
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
