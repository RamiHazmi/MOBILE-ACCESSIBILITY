// mobile/lib/handwriting_screen.dart — v4
// ═══════════════════════════════════════════════════════════════
// FLOW:
//   1. Screen opens → TTS speaks short guide in user's language
//   2. Live camera + framing guidance (spoken step-by-step)
//   3. Server says "Good" → badge turns green → "Tap to capture"
//   4. User TAPS → live feed freezes on that frame → spinner shown
//   5. Server extracts text → result spoken + shown (full, scrollable)
//   6. ADDRESS → navigate directly → back to home
//   7. Other → tap anywhere or 8 s auto-return → home
// ═══════════════════════════════════════════════════════════════

import 'dart:async';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image/image.dart' as img_lib;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as ws_status;

import 'api_service.dart';
import 'navigation_screen.dart';

// ── WS URL ───────────────────────────────────────────────────
String get _kWsUrl =>
    ApiService.baseUrl.replaceFirst(RegExp(r'^http'), 'ws') +
    '/ws/handwriting';

const int _kFps     = 4;
const int _kFrameMs = 1000 ~/ _kFps;

const _kBg      = Color(0xFF0A0A0F);
const _kPrimary = Color(0xFF7C6AF5);
const _kCard    = Color(0xFF1A1A2E);
const _kGreen   = Color(0xFF00AA55);

// ── TTS language map ─────────────────────────────────────────
const Map<String, String> _kTtsLang = {
  'english': 'en-US', 'french': 'fr-FR',
  'arabic':  'ar-SA', 'darija': 'ar-SA',
  'auto':    'en-US',
};

// ── Opening guide — one sentence, per language ──────────────
const Map<String, String> _kGuide = {
  'en-US': 'Point the camera at the document, then tap the screen to read.',
  'fr-FR': 'Pointez la caméra vers le document, puis appuyez pour lire.',
  'ar-SA': 'وجّه الكاميرا نحو المستند، ثم اضغط على الشاشة للقراءة.',
};

// ── Translate server guidance messages to user language ──────
// Server always sends English guidance. Translate for TTS.
const Map<String, Map<String, String>> _kGuidanceTr = {
  'fr-FR': {
    'Too dark. Move to better lighting.':
        'Trop sombre. Déplacez-vous vers une meilleure lumière.',
    'No text detected. Point camera at the document.':
        'Aucun texte détecté. Pointez la caméra vers le document.',
    'Good. Tap the screen to read.':
        'Bien. Appuyez sur l\'écran pour lire.',
  },
  'ar-SA': {
    'Too dark. Move to better lighting.':
        'الإضاءة ضعيفة. انتقل إلى مكان أكثر إضاءة.',
    'No text detected. Point camera at the document.':
        'لا يوجد نص. وجّه الكاميرا نحو المستند.',
    'Good. Tap the screen to read.':
        'ممتاز. اضغط على الشاشة للقراءة.',
  },
};

// ═══════════════════════════════════════════════════════════════

class HandwritingScreen extends StatefulWidget {
  final String language;
  const HandwritingScreen({super.key, this.language = 'auto'});

  @override
  State<HandwritingScreen> createState() => _HandwritingScreenState();
}

class _HandwritingScreenState extends State<HandwritingScreen>
    with WidgetsBindingObserver {

  // ── Camera ──────────────────────────────────────────────────
  CameraController? _camCtrl;
  bool   _camReady    = false;
  bool   _camError    = false;
  String _camErrMsg   = '';
  bool   _streamRunning = false;
  int    _lastFrameMs   = 0;

  // ── WebSocket ───────────────────────────────────────────────
  WebSocketChannel?   _channel;
  StreamSubscription? _wsSub;
  bool _wsConnected = false;
  int  _wsRetry     = 0;
  static const int _kMaxRetry = 5;

  // ── State ───────────────────────────────────────────────────
  String  _guidance   = 'Connecting…';
  bool    _stable     = false;   // server says framing is good
  bool    _capturing  = false;   // OCR running
  bool    _guideSpoken = false;
  Map<String, dynamic>? _result;

  // Frozen frame shown while OCR runs (replaces live feed)
  Uint8List? _frozenFrame;

  // Auto-return timer after result shown
  Timer?  _returnTimer;

  // ── TTS ─────────────────────────────────────────────────────
  final FlutterTts _tts = FlutterTts();
  String _lastSpoken = '';

  // ── Lifecycle ───────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initTts();
    _initCamera();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState s) {
    if (s == AppLifecycleState.inactive) {
      _stopStream();
      _camCtrl?.dispose();
      _camCtrl  = null;
      _camReady = false;
    } else if (s == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  @override
  void dispose() {
    _returnTimer?.cancel();
    WidgetsBinding.instance.removeObserver(this);
    _disconnect();
    _stopStream();
    _camCtrl?.dispose();
    _tts.stop();
    super.dispose();
  }

  // ── TTS ─────────────────────────────────────────────────────
  Future<void> _initTts() async {
    await _tts.setVolume(1.0);
    await _tts.setSpeechRate(0.50);
    await _tts.setPitch(1.0);
    await _tts.setLanguage('en-US');
    // Short guide — one sentence only
    Future<void>.delayed(const Duration(milliseconds: 600)).then((_) {
      if (mounted && !_guideSpoken) {
        _guideSpoken = true;
        final ttsLang = _kTtsLang[widget.language] ?? 'en-US';
        final guide   = _kGuide[ttsLang] ?? _kGuide['en-US']!;
        _speak(guide, lang: widget.language);
      }
    });
  }

  Future<void> _speak(String text, {String lang = 'auto'}) async {
    if (text.trim().isEmpty) return;
    final ttsLang = _kTtsLang[lang] ?? 'en-US';
    await _tts.setLanguage(ttsLang);
    await _tts.speak(text);
  }

  /// Translate server guidance (always English) to user's language, then speak.
  Future<void> _speakGuidance(String text) async {
    if (text.trim().isEmpty) return;
    final ttsLang = _kTtsLang[widget.language] ?? 'en-US';
    final tr      = _kGuidanceTr[ttsLang];
    final toSpeak = (tr != null && tr.containsKey(text)) ? tr[text]! : text;
    if (toSpeak == _lastSpoken) return;
    _lastSpoken = toSpeak;
    await _tts.setLanguage(ttsLang);
    await _tts.speak(toSpeak);
  }

  // ── Camera ──────────────────────────────────────────────────
  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw Exception('No cameras found');
      final cam = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final ctrl = CameraController(
        cam, ResolutionPreset.medium,
        enableAudio: false,
      );
      await ctrl.initialize();
      if (!mounted) return;
      setState(() { _camCtrl = ctrl; _camReady = true; _camError = false; });
      _connectWs();
    } catch (e) {
      if (!mounted) return;
      setState(() { _camError = true; _camErrMsg = e.toString(); });
    }
  }

  // ── WebSocket ───────────────────────────────────────────────
  Future<void> _connectWs() async {
    await Future<void>.delayed(const Duration(milliseconds: 300));
    if (!mounted) return;
    try {
      _channel = WebSocketChannel.connect(Uri.parse(_kWsUrl));
      await _channel!.ready.catchError((_) {});
      if (!mounted) return;
      _wsSub = _channel!.stream.listen(
        _onWsMessage, onError: _onWsError,
        onDone: _onWsDone, cancelOnError: false,
      );
      _wsRetry = 0;
      setState(() {
        _wsConnected = true;
        _guidance    = 'Ready. Point at document.';
      });
      _channel!.sink.add('SET_LANG:${widget.language}');
      Future<void>.delayed(const Duration(seconds: 2)).then((_) {
        if (mounted && !_streamRunning) _startStream();
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _guidance = 'Cannot connect to server.');
      _scheduleReconnect();
    }
  }

  void _disconnect() {
    _wsSub?.cancel();
    _channel?.sink.close(ws_status.goingAway);
    _wsConnected = false;
  }

  void _onWsError(Object _) {
    if (!mounted) return;
    setState(() { _wsConnected = false; _guidance = 'Connection lost…'; });
    _scheduleReconnect();
  }

  void _onWsDone() {
    if (!mounted) return;
    setState(() => _wsConnected = false);
    _scheduleReconnect();
  }

  void _scheduleReconnect() {
    if (!mounted || _wsRetry >= _kMaxRetry) return;
    _wsRetry++;
    Future<void>.delayed(Duration(seconds: _wsRetry)).then((_) {
      if (mounted && !_wsConnected) { _wsSub?.cancel(); _connectWs(); }
    });
  }

  // ── WS message handler ───────────────────────────────────────
  void _onWsMessage(dynamic raw) {
    if (raw is! String) return;
    final Map<String, dynamic> msg;
    try { msg = jsonDecode(raw) as Map<String, dynamic>; }
    catch (_) { return; }
    if (!mounted) return;

    final type = msg['type'] as String? ?? '';

    if (type == 'ready') {
      setState(() => _guidance = 'Point camera at the document.');
      _startStream();

    } else if (type == 'guide') {
      final guidance = msg['message'] as String? ?? '';
      final stable   = msg['stable'] == true;
      setState(() { _guidance = guidance; _stable = stable; });
      _speakGuidance(guidance);   // translated to user's language

    } else if (type == 'capture') {
      setState(() {
        _capturing = true;
        _guidance  = 'Reading…';
        _stable    = false;
      });

    } else if (type == 'result') {
      _handleResult(msg);

    } else if (type == 'error') {
      final errMsg = msg['message'] as String? ?? 'Error';
      setState(() { _capturing = false; _guidance = '⚠ $errMsg'; });
      _speak('Error. $errMsg');
    }
  }

  // ── Result handler ───────────────────────────────────────────
  void _handleResult(Map<String, dynamic> msg) {
    final spoken = msg['spoken_response'] as String? ?? '';
    final lang   = msg['language'] as String? ?? 'auto';
    final navTo  = msg['navigate_to'] as String? ?? '';
    final intent = msg['intent'] as String? ?? '';

    setState(() {
      _capturing   = false;
      _frozenFrame = null;
      _result      = msg;
      _guidance    = 'Done. Tap to go back.';
      _stable      = false;
    });

    if (spoken.isNotEmpty) {
      _lastSpoken = '';
      // Speak first, then act after TTS finishes
      _speakThenAct(spoken, lang: lang, onDone: () {
        if (!mounted) return;
        if (navTo.isNotEmpty && (intent == 'ADDRESS' || navTo.length > 4)) {
          // ADDRESS → navigate automatically after speech ends
          _goToNavigation(navTo, lang);
        } else {
          // Non-address → wait for tap, no auto-return timer
          // User taps anywhere to go back (handled in _onTap)
        }
      });
    } else {
      if (navTo.isNotEmpty && (intent == 'ADDRESS' || navTo.length > 4)) {
        _goToNavigation(navTo, lang);
      }
    }
  }

  /// Speak text and call [onDone] when TTS finishes.
  Future<void> _speakThenAct(String text,
      {String lang = 'auto', required VoidCallback onDone}) async {
    final ttsLang = _kTtsLang[lang] ?? 'en-US';
    await _tts.setLanguage(ttsLang);

    // Listen for completion
    _tts.setCompletionHandler(() {
      if (mounted) onDone();
    });
    // Also set a fallback timer based on word count in case
    // the completion handler doesn't fire (some Android TTS engines)
    final words    = text.trim().split(RegExp(r'\s+')).length;
    final fallback = Duration(milliseconds: (words * 280).clamp(3000, 60000));
    final timer    = Timer(fallback, () {
      if (mounted) onDone();
    });

    await _tts.speak(text);
    // fallback timer fires automatically if completion handler doesn't
  }

  void _goHome() {
    _returnTimer?.cancel();
    if (mounted) Navigator.of(context).pop();
  }

  void _goToNavigation(String address, String lang) {
    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (_) => NavigationScreen(
          destination: address,
          navMode: 'walking',
          language: lang,
        ),
      ),
    ).then((_) {
      ApiService.reset();
      if (mounted) Navigator.of(context).pop();
    });
  }

  // ── Frame streaming ──────────────────────────────────────────
  // Keep the latest encoded frame so we can freeze it on tap
  Uint8List? _latestFrameBytes;

  void _startStream() {
    if (_camCtrl == null || !_camCtrl!.value.isInitialized) return;
    if (_streamRunning) return;
    try { _camCtrl!.startImageStream(_onCameraImage); _streamRunning = true; }
    catch (_) {}
  }

  void _stopStream() {
    if (!_streamRunning) return;
    _streamRunning = false;
    _camCtrl?.stopImageStream().catchError((_) {});
  }

  void _onCameraImage(CameraImage image) {
    if (!_wsConnected || _channel == null || _capturing || !mounted) return;
    final now = DateTime.now().millisecondsSinceEpoch;
    if (now - _lastFrameMs < _kFrameMs) return;
    _lastFrameMs = now;

    // Capture sensor rotation once — used to correct YUV frames
    final sensorRotation =
        _camCtrl?.description.sensorOrientation ?? 0;

    _encodeFrame(image, sensorRotation).then((bytes) {
      if (bytes == null || !mounted) return;
      _latestFrameBytes = bytes;
      if (!_wsConnected || _channel == null || _capturing) return;
      try { _channel!.sink.add(bytes); } catch (_) {}
    });
  }

  Future<Uint8List?> _encodeFrame(CameraImage image, int sensorRotation) async {
    try {
      if (image.format.group == ImageFormatGroup.jpeg) {
        // Native JPEG already has correct orientation from the camera driver
        return image.planes[0].bytes;
      }
      // YUV420 — convert and apply sensor rotation so the image is upright
      return await compute(
        _yuvToJpeg,
        _YuvData.fromCameraImage(image, sensorRotation),
      );
    } catch (_) { return null; }
  }

  // ── Tap handler ──────────────────────────────────────────────
  void _onTap() {
    // Result showing → stop TTS and go home
    if (_result != null) {
      _tts.stop();
      _goHome();
      return;
    }
    // Already capturing → ignore
    if (_capturing) return;
    // Need a frame to work with
    if (_latestFrameBytes == null && !_wsConnected) return;

    // Freeze the live feed — show the last captured frame as a still
    final frozen = _latestFrameBytes;
    setState(() {
      _frozenFrame = frozen;
      _capturing   = true;
      _guidance    = 'Reading…';
      _stable      = false;
    });
    _stopStream();   // stop live feed — we have our frame

    // Tell server to capture now
    if (_wsConnected && _channel != null) {
      _channel!.sink.add('CAPTURE_NOW');
    }
  }

  // ═══════════════════════════════════════════════════════════
  // UI
  // ═══════════════════════════════════════════════════════════
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: _onTap,
      child: Scaffold(
        backgroundColor: _kBg,
        body: SafeArea(
          child: Stack(
            fit: StackFit.expand,
            children: [
              _buildCameraPreview(),

              // Top bar
              Positioned(
                top: 0, left: 0, right: 0,
                child: _buildTopBar(),
              ),

              // Guidance badge
              if (_result == null)
                Positioned(
                  bottom: 100, left: 16, right: 16,
                  child: _buildGuidanceBadge(),
                ),

              // Tap-to-capture hint (shown when stable, no result yet)
              if (_stable && !_capturing && _result == null)
                Positioned(
                  bottom: 48, left: 0, right: 0,
                  child: Center(
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 20, vertical: 10),
                      decoration: BoxDecoration(
                        color: _kGreen.withOpacity(0.85),
                        borderRadius: BorderRadius.circular(99),
                        boxShadow: const [
                          BoxShadow(color: Colors.black38, blurRadius: 8)
                        ],
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.touch_app_rounded,
                              color: Colors.white, size: 18),
                          SizedBox(width: 8),
                          Text('Tap to capture',
                              style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 15,
                                  fontWeight: FontWeight.w700)),
                        ],
                      ),
                    ),
                  ),
                ),

              // Result card
              if (_result != null)
                Positioned(
                  bottom: 0, left: 0, right: 0,
                  child: _buildResultCard(),
                ),

              // Capturing overlay — shown over frozen frame
              if (_capturing)
                Container(
                  color: Colors.black.withOpacity(0.60),
                  child: Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const CircularProgressIndicator(
                            color: _kPrimary, strokeWidth: 3),
                        const SizedBox(height: 20),
                        const Text('Extracting text…',
                            style: TextStyle(color: Colors.white,
                                fontSize: 18,
                                fontWeight: FontWeight.w600)),
                        const SizedBox(height: 8),
                        Text(_guidance,
                            style: TextStyle(
                                color: Colors.white.withOpacity(0.55),
                                fontSize: 13)),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (_camError) {
      return Container(color: _kBg,
        child: Center(child: Text(_camErrMsg,
            style: const TextStyle(color: Colors.red, fontSize: 13),
            textAlign: TextAlign.center)));
    }
    // Show frozen frame while OCR is running
    if (_frozenFrame != null) {
      return Image.memory(
        _frozenFrame!,
        fit: BoxFit.cover,
        width: double.infinity,
        height: double.infinity,
      );
    }
    if (!_camReady || _camCtrl == null) {
      return const Center(child: CircularProgressIndicator(color: _kPrimary));
    }
    return CameraPreview(_camCtrl!);
  }

  Widget _buildTopBar() {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 10, 16, 10),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter, end: Alignment.bottomCenter,
          colors: [Colors.black87, Colors.transparent],
        ),
      ),
      child: Row(children: [
        GestureDetector(
          onTap: _goHome,
          child: Container(
            width: 38, height: 38,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.15),
              borderRadius: BorderRadius.circular(999),
            ),
            child: const Icon(Icons.arrow_back_rounded,
                color: Colors.white, size: 20),
          ),
        ),
        const SizedBox(width: 12),
        const Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Read Document',
                  style: TextStyle(color: Colors.white,
                      fontSize: 18, fontWeight: FontWeight.w800)),
              Text('Tap to capture',
                  style: TextStyle(color: Colors.white70, fontSize: 11)),
            ]),
        ),
        // Connection dot
        Container(
          width: 10, height: 10,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: _wsConnected ? const Color(0xFF00E887) : Colors.red,
          ),
        ),
      ]),
    );
  }

  Widget _buildGuidanceBadge() {
    final Color badgeColor = _capturing ? _kPrimary
        : _stable ? _kGreen
        : Colors.black54;
    return Center(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: badgeColor,
          borderRadius: BorderRadius.circular(99),
          boxShadow: const [BoxShadow(color: Colors.black38, blurRadius: 8)],
        ),
        child: Row(mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_capturing)
              const SizedBox(width: 14, height: 14,
                child: CircularProgressIndicator(
                    strokeWidth: 2, color: Colors.white))
            else
              Icon(
                _stable ? Icons.check_circle_outline
                        : Icons.center_focus_strong,
                color: Colors.white, size: 16,
              ),
            const SizedBox(width: 8),
            Flexible(child: Text(_guidance,
                style: const TextStyle(color: Colors.white,
                    fontSize: 14, fontWeight: FontWeight.w600),
                textAlign: TextAlign.center)),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    final result       = _result!;
    final spoken       = result['spoken_response'] as String? ?? '';
    final intent       = result['intent'] as String? ?? 'PLAIN_TEXT';
    final lang         = result['language'] as String? ?? 'auto';
    final conf         = ((result['confidence'] as num?)?.toDouble() ?? 0) * 100;
    final hasIllegible = result['has_illegible'] as bool? ?? false;
    final isRtl        = lang == 'arabic' || lang == 'darija';
    final navTo        = result['navigate_to'] as String? ?? '';

    const Map<String, String> intentIcon = {
      'QUESTION': '❓', 'PLAIN_TEXT': '📄',
      'PRESCRIPTION': '💊', 'ADDRESS': '📍', 'FORM': '📋',
    };
    final icon = intentIcon[intent] ?? '📄';

    return Container(
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(context).size.height * 0.72,
        minHeight: 120,
      ),
      decoration: BoxDecoration(
        color: _kCard,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
        boxShadow: const [BoxShadow(color: Colors.black54, blurRadius: 16)],
      ),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        // Drag handle
        Container(
          margin: const EdgeInsets.only(top: 10),
          width: 36, height: 4,
          decoration: BoxDecoration(
            color: Colors.white24,
            borderRadius: BorderRadius.circular(2),
          ),
        ),
        // Header row
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 10, 16, 0),
          child: Row(children: [
            Text('$icon  ${intent.replaceAll('_', ' ')}',
                style: const TextStyle(color: _kPrimary,
                    fontWeight: FontWeight.bold, fontSize: 13)),
            const Spacer(),
            Text('${conf.round()}%',
                style: TextStyle(
                    color: conf >= 70 ? _kGreen : Colors.orange,
                    fontSize: 12, fontWeight: FontWeight.w700)),
            const SizedBox(width: 12),
            GestureDetector(
              onTap: () => _speak(spoken, lang: lang),
              child: const Icon(Icons.volume_up_rounded,
                  color: _kPrimary, size: 22),
            ),
          ]),
        ),
        if (hasIllegible)
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 6, 16, 0),
            child: Row(children: const [
              Icon(Icons.warning_amber, color: Colors.redAccent, size: 14),
              SizedBox(width: 6),
              Expanded(child: Text(
                'Some parts unclear — confirm with pharmacist.',
                style: TextStyle(color: Colors.redAccent, fontSize: 11))),
            ]),
          ),
        // Navigate button if address
        if (navTo.isNotEmpty)
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
            child: SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: () => _goToNavigation(navTo, lang),
                icon: const Icon(Icons.navigation_rounded, size: 16),
                label: Text('Navigate: $navTo',
                    maxLines: 1, overflow: TextOverflow.ellipsis),
                style: ElevatedButton.styleFrom(
                  backgroundColor: _kPrimary,
                  foregroundColor: Colors.white,
                  textStyle: const TextStyle(fontSize: 12),
                  padding: const EdgeInsets.symmetric(
                      vertical: 8, horizontal: 12),
                ),
              ),
            ),
          ),
        // Full text — scrollable
        Flexible(
          child: Scrollbar(
            thumbVisibility: true,
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(16, 10, 16, 20),
              child: Directionality(
                textDirection:
                    isRtl ? TextDirection.rtl : TextDirection.ltr,
                child: SelectableText(
                  spoken,
                  style: const TextStyle(
                      color: Colors.white, fontSize: 15, height: 1.7),
                ),
              ),
            ),
          ),
        ),
        // Tap-to-go-home hint
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 14),
          child: Text(
            'Tap anywhere to go back',
            style: TextStyle(
                color: Colors.white.withOpacity(0.35), fontSize: 12),
            textAlign: TextAlign.center,
          ),
        ),
      ]),
    );
  }
}

// ── YUV420 → JPEG (background isolate) ────────────────────────
class _YuvData {
  final int width, height;
  final Uint8List yPlane, uPlane, vPlane;
  final int yRowStride, uvRowStride, uvPixelStride;
  final int sensorRotation; // 0, 90, 180, 270

  const _YuvData({
    required this.width, required this.height,
    required this.yPlane, required this.uPlane, required this.vPlane,
    required this.yRowStride, required this.uvRowStride,
    required this.uvPixelStride,
    this.sensorRotation = 0,
  });

  factory _YuvData.fromCameraImage(CameraImage im, [int rotation = 0]) =>
      _YuvData(
        width: im.width, height: im.height,
        yPlane: im.planes[0].bytes,
        uPlane: im.planes[1].bytes,
        vPlane: im.planes[2].bytes,
        yRowStride:    im.planes[0].bytesPerRow,
        uvRowStride:   im.planes[1].bytesPerRow,
        uvPixelStride: im.planes[1].bytesPerPixel ?? 1,
        sensorRotation: rotation,
      );
}

Uint8List? _yuvToJpeg(_YuvData d) {
  try {
    // Build raw RGB image from YUV planes
    final src = img_lib.Image(width: d.width, height: d.height);
    for (int y = 0; y < d.height; y++) {
      for (int x = 0; x < d.width; x++) {
        final yVal  = d.yPlane[y * d.yRowStride + x] & 0xFF;
        final uvRow = y >> 1;
        final uvCol = (x >> 1) * d.uvPixelStride;
        final uVal  = d.uPlane[uvRow * d.uvRowStride + uvCol] & 0xFF;
        final vVal  = d.vPlane[uvRow * d.uvRowStride + uvCol] & 0xFF;
        final r = (yVal + 1.402   * (vVal - 128)).round().clamp(0, 255);
        final g = (yVal - 0.34414 * (uVal - 128) -
                   0.71414 * (vVal - 128)).round().clamp(0, 255);
        final b = (yVal + 1.772   * (uVal - 128)).round().clamp(0, 255);
        src.setPixelRgb(x, y, r, g, b);
      }
    }

    // Apply sensor rotation so the image is upright regardless of
    // how the camera sensor is physically mounted in the device.
    img_lib.Image oriented;
    switch (d.sensorRotation) {
      case 90:
        oriented = img_lib.copyRotate(src, angle: 90);
        break;
      case 180:
        oriented = img_lib.copyRotate(src, angle: 180);
        break;
      case 270:
        oriented = img_lib.copyRotate(src, angle: 270);
        break;
      default:
        oriented = src; // 0° — already upright
    }

    return Uint8List.fromList(img_lib.encodeJpg(oriented, quality: 75));
  } catch (_) { return null; }
}
