// mobile/lib/sign_language_page.dart
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image/image.dart' as img;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as ws_status;

import 'api_service.dart';
import 'text_to_sign_page.dart';

String get _kWsUrl =>
    ApiService.baseUrl.replaceFirst(RegExp(r'^http'), 'ws') + '/ws/gesture';

const _kPrimary   = Color(0xFF2E8BFF);
const _kAccent    = Color(0xFF00C896);
const _kSecondary = Color(0xFF4FC3F7);
const _kBg        = Color(0xFF060E22);
const _kCard      = Color(0xFF0D1B3E);
const _kText      = Colors.white;
const _kSubtext   = Color(0xFF8BA0C8);

/// Target **sleep** time between first and last frame (gaps only; `takePicture` adds more).
const int _kTargetCaptureSpanMs = 5000;  // 5 seconds — spread T_MAX frames evenly

enum _Mode { word, letter, sentence }

class SignLanguagePage extends StatefulWidget {
  const SignLanguagePage({super.key});

  @override
  State<SignLanguagePage> createState() => _SignLanguagePageState();
}

class _SignLanguagePageState extends State<SignLanguagePage>
    with WidgetsBindingObserver {

  CameraController? _cam;
  bool _camReady = false;
  bool _camError = false;

  WebSocketChannel?   _channel;
  StreamSubscription? _wsSub;
  bool _wsConnected = false;

  final FlutterTts _tts = FlutterTts();

  _Mode  _mode       = _Mode.letter;
  bool   _serviceOk  = false;
  bool   _capturing  = false;
  bool   _building   = false;
  String _statusMsg  = 'Connecting…';

  /// From server `ready` → `t_max` (same as model `config.json` T_MAX). Default until first message.
  int _tMax = 60;

  String  _lastSign      = '';
  String  _lastEnglish   = '';
  String  _lastHandshape = '';
  double  _lastConf      = 0.0;
  List<String> _lastAlts = [];

  final List<String> _sentenceSigns = [];
  String _builtSentence = '';
  double _sentenceConf  = 0.0;

  final List<_SignResult> _history = [];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initTts();
    _initCamera();
  }

  Future<void> _initTts() async {
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.45);
    await _tts.setVolume(1.0);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) _stopCamera();
    else if (state == AppLifecycleState.resumed) _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _disconnect();
    _cam?.dispose();
    _tts.stop();
    super.dispose();
  }

  Future<void> _initCamera() async {
    try {
      final cams = await availableCameras();
      final cam  = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );
      final ctrl = CameraController(
        cam, ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await ctrl.initialize();
      if (!mounted) return;
      setState(() { _cam = ctrl; _camReady = true; _camError = false; });
      _connectWs();
    } catch (e) {
      if (!mounted) return;
      setState(() { _camError = true; _statusMsg = 'Camera error: $e'; });
    }
  }

  void _stopCamera() { _cam?.dispose(); _cam = null; _camReady = false; }

  void _connectWs() {
    try {
      _channel = WebSocketChannel.connect(Uri.parse(_kWsUrl));
      _wsSub   = _channel!.stream.listen(
          _onMsg, onError: _onWsError, onDone: _onWsDone);
      setState(() { _wsConnected = true; _statusMsg = 'Connected'; });
    } catch (_) {
      setState(() {
        _wsConnected = false;
        _statusMsg = 'Cannot connect — check Wi-Fi & PC IP';
      });
    }
  }

  void _disconnect() {
    _wsSub?.cancel();
    _channel?.sink.close(ws_status.goingAway);
    _wsConnected = false;
  }

  void _onMsg(dynamic raw) {
    if (raw is! String) return;
    Map<String, dynamic> msg;
    try { msg = jsonDecode(raw) as Map<String, dynamic>; }
    catch (_) { return; }
    if (!mounted) return;

    final type = msg['type'] as String? ?? '';

    setState(() {
      if (type == 'ready') {
        _serviceOk = msg['ok'] == true;
        final tm = msg['t_max'];
        if (tm != null) {
          final n = (tm as num).toInt();
          if (n > 0) _tMax = n.clamp(1, 512);
        }
        _statusMsg = _serviceOk
            ? 'Ready — show hand centre-frame, tap Capture ($_tMax frames)'
            : 'ASL CNN not loaded — check models/sign/new/asl_cnn.pt';

      } else if (type == 'status') {
        _statusMsg = msg['message'] as String? ?? '';

      } else if (type == 'letter') {
        _capturing  = false;
        final letter = msg['letter'] as String? ?? '?';
        final conf   = (msg['confidence'] as num?)?.toDouble() ?? 0;
        final alt    = msg['alternative'] as String?;
        _lastSign      = letter;
        _lastEnglish   = 'Letter $letter';
        _lastConf      = conf;
        _lastHandshape = '';
        _lastAlts      = alt != null ? [alt] : [];
        _statusMsg     = 'Letter recognised';
        if (_mode == _Mode.sentence) _sentenceSigns.add(letter);
        final speak = (alt != null && conf < 0.7) ? '$letter — or maybe $alt' : letter;
        _tts.speak(speak);
        _addHistory(_SignResult(sign: letter, english: 'Letter $letter',
            confidence: conf, mode: _Mode.letter));

      } else if (type == 'word') {
        _capturing  = false;
        final sign    = msg['sign']    as String? ?? 'UNCLEAR';
        final english = msg['english'] as String? ?? '';
        final conf    = (msg['confidence'] as num?)?.toDouble() ?? 0;
        final hs      = msg['handshape'] as String? ?? '';
        final alts    = (msg['alternatives'] as List<dynamic>?)
            ?.map((e) => e.toString()).toList() ?? [];
        _lastSign      = sign;
        _lastEnglish   = english;
        _lastConf      = conf;
        _lastHandshape = hs;
        _lastAlts      = alts;
        _statusMsg     = english.isEmpty ? 'Unclear — try again' : 'Sign recognised';
        if (_mode == _Mode.sentence && sign != 'UNCLEAR' && english.isNotEmpty) {
          _sentenceSigns.add(sign);
        }
        if (english.isNotEmpty) _tts.speak(english);
        _addHistory(_SignResult(sign: sign, english: english,
            confidence: conf, mode: _Mode.word));

      } else if (type == 'sentence') {
        _building      = false;
        _builtSentence = msg['sentence'] as String? ?? '';
        _sentenceConf  = (msg['confidence'] as num?)?.toDouble() ?? 0;
        _statusMsg     = 'Sentence built';
        if (_builtSentence.isNotEmpty) _tts.speak(_builtSentence);

      } else if (type == 'error') {
        _capturing = false;
        _building  = false;
        _statusMsg = '⚠ ${msg['message'] ?? 'Error'}';
      }
    });
  }

  void _onWsError(Object _) {
    if (!mounted) return;
    setState(() { _wsConnected = false; _statusMsg = 'Connection lost'; });
  }

  void _onWsDone() {
    if (!mounted) return;
    setState(() { _wsConnected = false; _statusMsg = 'Disconnected'; });
  }

  void _addHistory(_SignResult r) {
    _history.insert(0, r);
    if (_history.length > 30) _history.removeLast();
  }

  /// Milliseconds to wait after each frame so the **delays** sum to ~[_kTargetCaptureSpanMs]
  /// (e.g. 150 frames → 149 gaps → ≈33 ms, ≈5 s of spacing only).
  int _frameGapMs(int framesToCapture) {
    if (framesToCapture <= 1) return 0;
    return (_kTargetCaptureSpanMs / (framesToCapture - 1))
        .round()
        .clamp(0, 60000);
  }

  /// One full sequence: T_MAX still images (matches server model window).
  Future<void> _capture() async {
    if (!_wsConnected || !_serviceOk || _capturing || _building) return;
    if (_cam == null || !_cam!.value.isInitialized) return;
    final framesToCapture = _tMax.clamp(1, 512);  // use full T_MAX, no hard cap
    setState(() {
      _capturing = true;
      _statusMsg = 'Capturing $_tMax frames over 5 s…';
    });
    try {
      final framesB64 = await _captureStreamFrames(
        framesToCapture: framesToCapture,
        spanMs: _kTargetCaptureSpanMs,
      );
      if (framesB64.isEmpty) {
        setState(() {
          _capturing = false;
          _statusMsg = 'No frames captured from stream — try again';
        });
        return;
      }

      final modeStr = _mode == _Mode.letter ? 'letter' : 'word';
      _channel!.sink.add(jsonEncode({
        'cmd': 'INTERPRET',
        'mode': modeStr,
        'frames_b64': framesB64,   // multi-frame — server picks up motion
      }));
    } catch (e) {
      setState(() { _capturing = false; _statusMsg = 'Capture failed: $e'; });
    }
  }

  Future<List<String>> _captureStreamFrames({
    required int framesToCapture,
    required int spanMs,
  }) async {
    final ctrl = _cam;
    if (ctrl == null || !ctrl.value.isInitialized) return [];

    if (ctrl.value.isTakingPicture) return [];
    if (ctrl.value.isStreamingImages) {
      await ctrl.stopImageStream();
    }

    final out = <String>[];
    final completer = Completer<List<String>>();
    final sampleGapMs = _frameGapMs(framesToCapture);
    DateTime lastSample = DateTime.fromMillisecondsSinceEpoch(0);
    bool encoding = false;

    Future<void> stopAndComplete() async {
      if (completer.isCompleted) return;
      try {
        if (ctrl.value.isStreamingImages) {
          await ctrl.stopImageStream();
        }
      } catch (_) {}
      completer.complete(out);
    }

    final timer = Timer(Duration(milliseconds: spanMs), () async {
      await stopAndComplete();
    });

    await ctrl.startImageStream((CameraImage image) async {
      if (completer.isCompleted || encoding) return;
      if (out.length >= framesToCapture) {
        await stopAndComplete();
        return;
      }
      final now = DateTime.now();
      if (now.difference(lastSample).inMilliseconds < sampleGapMs) return;

      encoding = true;
      try {
        final jpeg = _cameraImageToJpeg(image);
        if (jpeg != null && jpeg.isNotEmpty) {
          out.add(base64Encode(jpeg));
          lastSample = now;
          if (mounted) {
            setState(() {
              _statusMsg = 'Frame ${out.length}/$framesToCapture captured…';
            });
          }
        }
      } finally {
        encoding = false;
      }

      if (out.length >= framesToCapture) {
        await stopAndComplete();
      }
    });

    final res = await completer.future;
    timer.cancel();
    return res;
  }

  Uint8List? _cameraImageToJpeg(CameraImage image) {
    img.Image? rgb;
    if (image.format.group == ImageFormatGroup.yuv420 &&
        image.planes.length >= 3) {
      rgb = _yuv420ToImage(image);
    } else if (image.format.group == ImageFormatGroup.bgra8888 &&
        image.planes.isNotEmpty) {
      rgb = _bgra8888ToImage(image);
    }
    if (rgb == null) return null;
    final jpg = img.encodeJpg(rgb, quality: 65);
    return Uint8List.fromList(jpg);
  }

  img.Image _yuv420ToImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final yBytes = yPlane.bytes;
    final uBytes = uPlane.bytes;
    final vBytes = vPlane.bytes;

    final yRowStride = yPlane.bytesPerRow;
    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;

    final out = img.Image(width: width, height: height);
    for (int y = 0; y < height; y++) {
      final yRow = y * yRowStride;
      final uvRow = (y >> 1) * uvRowStride;
      for (int x = 0; x < width; x++) {
        final yIndex = yRow + x;
        final uvIndex = uvRow + (x >> 1) * uvPixelStride;

        final yp = yBytes[yIndex];
        final up = uBytes[uvIndex];
        final vp = vBytes[uvIndex];

        int r = (yp + 1.402 * (vp - 128)).round();
        int g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128)).round();
        int b = (yp + 1.772 * (up - 128)).round();
        if (r < 0) r = 0;
        if (g < 0) g = 0;
        if (b < 0) b = 0;
        if (r > 255) r = 255;
        if (g > 255) g = 255;
        if (b > 255) b = 255;
        out.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return out;
  }

  img.Image _bgra8888ToImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final plane = image.planes[0];
    final bytes = plane.bytes;
    final rowStride = plane.bytesPerRow;
    final out = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      final rowStart = y * rowStride;
      for (int x = 0; x < width; x++) {
        final i = rowStart + x * 4;
        final b = bytes[i];
        final g = bytes[i + 1];
        final r = bytes[i + 2];
        final a = bytes[i + 3];
        out.setPixelRgba(x, y, r, g, b, a);
      }
    }
    return out;
  }

  void _buildSentence() {
    if (_sentenceSigns.isEmpty || _building) return;
    setState(() { _building = true; });
    _channel!.sink.add(jsonEncode({'cmd': 'BUILD_SENTENCE', 'signs': _sentenceSigns}));
  }

  void _clearSentence() {
    setState(() {
      _sentenceSigns.clear(); _builtSentence = ''; _sentenceConf = 0;
      _statusMsg = 'Cleared — start signing';
    });
  }

  void _reset() {
    _channel?.sink.add(jsonEncode({'cmd': 'RESET'}));
    setState(() {
      _lastSign = ''; _lastEnglish = ''; _lastHandshape = '';
      _lastConf = 0; _lastAlts = [];
      _statusMsg = 'Ready — show your hand and tap Capture';
    });
  }

  // ═══════════════════════════════════════════════════════════
  // BUILD
  // ═══════════════════════════════════════════════════════════
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _kBg,
      body: SafeArea(
        child: Column(children: [
          _buildHeader(context),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 14),
                  _buildStatusBar(),
                  const SizedBox(height: 10),
                  _buildTextToSignBanner(context),
                  const SizedBox(height: 14),
                  _buildCamera(),
                  const SizedBox(height: 14),
                  _buildCaptureButton(),
                  const SizedBox(height: 16),
                  if (_lastSign.isNotEmpty) ...[
                    _buildResultCard(),
                    const SizedBox(height: 14),
                  ],
                  if (_mode == _Mode.sentence) ...[
                    _buildSentenceBuilder(),
                    const SizedBox(height: 14),
                  ],
                  if (_history.isNotEmpty) _buildHistory(),
                ],
              ),
            ),
          ),
        ]),
      ),
    );
  }

  Widget _buildHeader(BuildContext ctx) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [_kPrimary, _kSecondary],
          begin: Alignment.topLeft, end: Alignment.bottomRight,
        ),
      ),
      child: Row(children: [
        GestureDetector(
          onTap: () => Navigator.of(ctx).pop(),
          child: Container(
            width: 38, height: 38,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(999),
            ),
            child: const Icon(Icons.arrow_back_rounded, color: Colors.white, size: 20),
          ),
        ),
        const SizedBox(width: 12),
        const Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Sign Language',
                style: TextStyle(color: Colors.white,
                    fontSize: 20, fontWeight: FontWeight.w800)),
            Text('ASL CNN · A–Y alphabet · fingerspelling',
                style: TextStyle(color: Color(0xFFE7ECFF), fontSize: 12)),
          ]),
        ),
        GestureDetector(
          onTap: () => Navigator.of(ctx).push(
            MaterialPageRoute(builder: (_) => const TextToSignPage()),
          ),
          child: Container(
            width: 38, height: 38,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(999),
            ),
            child: const Center(
              child: Text('✍️', style: TextStyle(fontSize: 18)),
            ),
          ),
        ),
        const SizedBox(width: 8),
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

  Widget _buildTextToSignBanner(BuildContext ctx) {
    return GestureDetector(
      onTap: () => Navigator.of(ctx).push(
        MaterialPageRoute(builder: (_) => const TextToSignPage()),
      ),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [_kAccent.withOpacity(0.15), _kPrimary.withOpacity(0.1)],
          ),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: _kAccent.withOpacity(0.35)),
        ),
        child: Row(children: [
          Container(
            width: 36, height: 36,
            decoration: BoxDecoration(
              color: _kAccent.withOpacity(0.18),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Center(
              child: Text('✍️', style: TextStyle(fontSize: 18)),
            ),
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Text → Sign Language',
                  style: TextStyle(color: _kText, fontSize: 13,
                      fontWeight: FontWeight.w800)),
              Text('Type a sentence, see it signed step by step',
                  style: TextStyle(color: _kSubtext, fontSize: 11)),
            ]),
          ),
          const Icon(Icons.arrow_forward_ios_rounded, color: _kAccent, size: 14),
        ]),
      ),
    );
  }

  Widget _buildModeSelector() {
    return Container(
      decoration: BoxDecoration(
        color: _kCard,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _kPrimary.withOpacity(0.15)),
      ),
      child: Row(children: [
        _ModeTab(label: '🤟 Word',     active: _mode == _Mode.word,
            onTap: () => setState(() { _mode = _Mode.word;     _reset(); })),
        _ModeTab(label: '🔤 Letter',   active: _mode == _Mode.letter,
            onTap: () => setState(() { _mode = _Mode.letter;   _reset(); })),
        _ModeTab(label: '💬 Sentence', active: _mode == _Mode.sentence,
            onTap: () => setState(() { _mode = _Mode.sentence; _reset(); })),
      ]),
    );
  }

  Widget _buildStatusBar() {
    final color = _capturing || _building ? _kAccent
        : _serviceOk ? _kPrimary : Colors.orange;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: _kCard,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color, width: 1.5),
      ),
      child: Row(children: [
        if (_capturing || _building)
          const SizedBox(width: 16, height: 16,
              child: CircularProgressIndicator(strokeWidth: 2, color: _kAccent))
        else
          Icon(_serviceOk ? Icons.check_circle_outline : Icons.warning_amber_rounded,
              color: color, size: 18),
        const SizedBox(width: 10),
        Expanded(child: Text(_statusMsg,
            style: const TextStyle(color: _kText,
                fontSize: 13, fontWeight: FontWeight.w500))),
      ]),
    );
  }

  Widget _buildCamera() {
    final guideText = _mode == _Mode.letter
        ? '✋\nPlace hand\nhere'
        : _mode == _Mode.word
            ? '🤟\nSign here'
            : '🤟\nSign one word\nat a time';

    return Container(
      height: 420,
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: _capturing ? _kAccent : _kPrimary.withOpacity(0.3),
          width: _capturing ? 2.5 : 1.0,
        ),
      ),
      clipBehavior: Clip.hardEdge,
      child: Stack(fit: StackFit.expand, children: [
        if (_camReady && _cam != null)
          CameraPreview(_cam!)
        else if (_camError)
          const Center(child: Text('Camera error',
              style: TextStyle(color: Colors.red)))
        else
          const Center(child: CircularProgressIndicator(color: _kPrimary)),
        // Guide box — fills most of the frame width
        Positioned(
          left: 16, right: 16, top: 16, bottom: 16,
          child: Container(
            decoration: BoxDecoration(
              border: Border.all(
                color: _capturing ? _kAccent : Colors.white.withOpacity(0.4),
                width: 1.5,
              ),
              borderRadius: BorderRadius.circular(16),
            ),
            child: _capturing
                ? null
                : Center(child: Text(guideText,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                        color: Colors.white.withOpacity(0.7),
                        fontSize: 14, height: 1.5))),
          ),
        ),
        if (_capturing) Container(color: _kAccent.withOpacity(0.08)),
      ]),
    );
  }

  Widget _buildCaptureButton() {
    final enabled = _wsConnected && _serviceOk && !_capturing && !_building;
    return GestureDetector(
      onTap: enabled ? _capture : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        height: 60,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: enabled
                ? [_kPrimary, _kAccent]
                : [Colors.grey.shade300, Colors.grey.shade300],
          ),
          borderRadius: BorderRadius.circular(16),
          boxShadow: enabled ? [BoxShadow(color: _kPrimary.withOpacity(0.35),
              blurRadius: 16, offset: const Offset(0, 6))] : [],
        ),
        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(_capturing ? Icons.hourglass_top_rounded : Icons.camera_alt_rounded,
              color: Colors.white, size: 22),
          const SizedBox(width: 10),
          Text(
            _capturing
                ? _statusMsg
                : 'Capture & Interpret ($_tMax frames · 5 s)',
              style: const TextStyle(color: Colors.white,
                  fontSize: 16, fontWeight: FontWeight.w800)),
        ]),
      ),
    );
  }

  Widget _buildResultCard() {
    final confColor = _lastConf >= 0.75 ? const Color(0xFF00AA55)
        : _lastConf >= 0.5 ? Colors.orange : Colors.red;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(colors: [
          _kPrimary.withOpacity(0.07), _kAccent.withOpacity(0.07),
        ]),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _kPrimary.withOpacity(0.25)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Icon(Icons.sign_language_rounded, color: _kPrimary, size: 18),
          const SizedBox(width: 6),
          Text(_mode == _Mode.letter ? 'Fingerspelling' : 'Sign recognised',
              style: const TextStyle(color: _kPrimary, fontSize: 12,
                  fontWeight: FontWeight.w700, letterSpacing: 0.5)),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(
              color: confColor.withOpacity(0.12),
              borderRadius: BorderRadius.circular(99),
            ),
            child: Text('${(_lastConf * 100).toStringAsFixed(0)}% conf',
                style: TextStyle(color: confColor,
                    fontSize: 11, fontWeight: FontWeight.w700)),
          ),
          const SizedBox(width: 8),
          GestureDetector(
            onTap: () => _tts.speak(
                _lastEnglish.isNotEmpty ? _lastEnglish : _lastSign),
            child: const Icon(Icons.volume_up_rounded, color: _kPrimary, size: 20),
          ),
        ]),
        const SizedBox(height: 12),
        Text(
          _mode == _Mode.letter ? _lastSign : _lastEnglish,
          style: const TextStyle(color: _kText,
              fontSize: 32, fontWeight: FontWeight.w900),
        ),
        if (_mode == _Mode.word && _lastSign.isNotEmpty) ...[
          const SizedBox(height: 4),
          Text('ASL: $_lastSign',
              style: const TextStyle(color: _kSubtext,
                  fontSize: 13, fontWeight: FontWeight.w600)),
        ],
        if (_lastHandshape.isNotEmpty) ...[
          const SizedBox(height: 8),
          Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Icon(Icons.back_hand_outlined, color: _kSubtext, size: 13),
            const SizedBox(width: 4),
            Expanded(child: Text(_lastHandshape,
                style: const TextStyle(color: _kSubtext, fontSize: 12))),
          ]),
        ],
        if (_lastAlts.isNotEmpty) ...[
          const SizedBox(height: 8),
          Wrap(spacing: 6, children: [
            const Text('Alt:', style: TextStyle(color: _kSubtext, fontSize: 11)),
            ..._lastAlts.map((a) => Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: _kPrimary.withOpacity(0.08),
                borderRadius: BorderRadius.circular(99),
              ),
              child: Text(a, style: const TextStyle(color: _kPrimary,
                  fontSize: 11, fontWeight: FontWeight.w600)),
            )),
          ]),
        ],
      ]),
    );
  }

  Widget _buildSentenceBuilder() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _kCard,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _kSecondary.withOpacity(0.25)),
        boxShadow: const [BoxShadow(
            color: Color(0x0F22335B), blurRadius: 12, offset: Offset(0, 4))],
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Icon(Icons.chat_bubble_outline_rounded,
              color: _kSecondary, size: 18),
          const SizedBox(width: 8),
          const Text('Sentence Builder',
              style: TextStyle(color: _kSecondary, fontSize: 13,
                  fontWeight: FontWeight.w700)),
          const Spacer(),
          GestureDetector(
            onTap: _clearSentence,
            child: const Icon(Icons.delete_outline_rounded,
                color: _kSubtext, size: 20),
          ),
        ]),
        const SizedBox(height: 12),
        if (_sentenceSigns.isEmpty)
          const Text('No signs yet — capture signs one by one.',
              style: TextStyle(color: _kSubtext, fontSize: 13))
        else
          Wrap(spacing: 6, runSpacing: 6,
            children: _sentenceSigns.asMap().entries.map((e) {
              return GestureDetector(
                onTap: () => setState(() => _sentenceSigns.removeAt(e.key)),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: _kSecondary.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(99),
                    border: Border.all(color: _kSecondary.withOpacity(0.3)),
                  ),
                  child: Row(mainAxisSize: MainAxisSize.min, children: [
                    Text(e.value, style: const TextStyle(color: _kSecondary,
                        fontSize: 13, fontWeight: FontWeight.w700)),
                    const SizedBox(width: 4),
                    const Icon(Icons.close_rounded, color: _kSecondary, size: 12),
                  ]),
                ),
              );
            }).toList()),
        const SizedBox(height: 12),
        GestureDetector(
          onTap: (_sentenceSigns.isNotEmpty && !_building) ? _buildSentence : null,
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(vertical: 12),
            decoration: BoxDecoration(
              color: (_sentenceSigns.isNotEmpty && !_building)
                  ? _kSecondary : Colors.grey.shade300,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              if (_building)
                const SizedBox(width: 16, height: 16,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white))
              else
                const Icon(Icons.auto_fix_high_rounded, color: Colors.white, size: 18),
              const SizedBox(width: 8),
              Text(
                _building
                    ? 'Building…'
                    : 'Build Sentence (${_sentenceSigns.length} signs)',
                style: const TextStyle(color: Colors.white,
                    fontSize: 14, fontWeight: FontWeight.w700),
              ),
            ]),
          ),
        ),
        if (_builtSentence.isNotEmpty) ...[
          const SizedBox(height: 14),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: _kAccent.withOpacity(0.08),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: _kAccent.withOpacity(0.3)),
            ),
            child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const Icon(Icons.format_quote_rounded, color: _kAccent, size: 18),
              const SizedBox(width: 8),
              Expanded(child: Text(_builtSentence,
                  style: const TextStyle(color: _kText,
                      fontSize: 16, fontWeight: FontWeight.w700, height: 1.4))),
              const SizedBox(width: 8),
              GestureDetector(
                onTap: () => _tts.speak(_builtSentence),
                child: const Icon(Icons.volume_up_rounded,
                    color: _kAccent, size: 20),
              ),
            ]),
          ),
          const SizedBox(height: 4),
          Text('Confidence: ${(_sentenceConf * 100).toStringAsFixed(0)}%',
              style: const TextStyle(color: _kSubtext, fontSize: 11)),
        ],
      ]),
    );
  }

  Widget _buildHistory() {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const Text('History', style: TextStyle(color: _kText,
          fontSize: 14, fontWeight: FontWeight.w700)),
      const SizedBox(height: 8),
      ..._history.take(8).map((r) => _HistoryTile(
          result: r,
          onSpeak: () => _tts.speak(r.english.isNotEmpty ? r.english : r.sign))),
    ]);
  }

  Widget _buildTips() { return const SizedBox.shrink(); }
}

// ═══════════════════════════════════════════════════════════════
// Sub-widgets
// ═══════════════════════════════════════════════════════════════

class _ModeTab extends StatelessWidget {
  final String label;
  final bool active;
  final VoidCallback onTap;
  const _ModeTab({required this.label, required this.active, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          margin: const EdgeInsets.all(4),
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: active ? _kPrimary : Colors.transparent,
            borderRadius: BorderRadius.circular(10),
          ),
          child: Text(label,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: active ? Colors.white : _kSubtext,
                fontSize: 13, fontWeight: FontWeight.w700,
              )),
        ),
      ),
    );
  }
}

class _HistoryTile extends StatelessWidget {
  final _SignResult result;
  final VoidCallback onSpeak;
  const _HistoryTile({required this.result, required this.onSpeak});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
      decoration: BoxDecoration(
        color: _kCard,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _kPrimary.withOpacity(0.12)),
      ),
      child: Row(children: [
        Icon(
          result.mode == _Mode.letter
              ? Icons.text_fields_rounded : Icons.sign_language_rounded,
          color: _kPrimary, size: 16,
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(result.english.isNotEmpty ? result.english : result.sign,
                style: const TextStyle(color: _kText,
                    fontSize: 14, fontWeight: FontWeight.w700)),
            if (result.english.isNotEmpty && result.sign != result.english)
              Text(result.sign,
                  style: const TextStyle(color: _kSubtext, fontSize: 11)),
          ]),
        ),
        Text('${(result.confidence * 100).toStringAsFixed(0)}%',
            style: TextStyle(
              color: result.confidence >= 0.7
                  ? const Color(0xFF00AA55) : Colors.orange,
              fontSize: 11, fontWeight: FontWeight.w700,
            )),
        const SizedBox(width: 8),
        GestureDetector(
          onTap: onSpeak,
          child: const Icon(Icons.volume_up_rounded, color: _kSubtext, size: 16),
        ),
      ]),
    );
  }
}

class _SignResult {
  final String sign;
  final String english;
  final double confidence;
  final _Mode  mode;
  const _SignResult({
    required this.sign,
    required this.english,
    required this.confidence,
    required this.mode,
  });
}