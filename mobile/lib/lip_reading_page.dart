// mobile/lib/lip_reading_page.dart
// ═══════════════════════════════════════════════════════════════
// Real-time lip reading page.
//
// Flow:
//   1. Open page  → camera preview + connect WebSocket
//   2. Tap Record → server starts buffering frames
//   3. Articulate words silently (English, BBC-style)
//   4. Tap Predict → server runs Chaplin → returns text
//   5. Text appears in big card; translation buttons + action cards shown
// ═══════════════════════════════════════════════════════════════

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as ws_status;

import 'api_service.dart';

// WS URL derived from ApiService.baseUrl
String get _kWsUrl =>
    ApiService.baseUrl.replaceFirst(RegExp(r'^http'), 'ws') + '/ws/vsr';

const int _kTargetFps       = 16;
const int _kFrameIntervalMs = 1000 ~/ _kTargetFps;

// Colors
const _kPrimary   = Color(0xFF4C6FFF);
const _kSecondary = Color(0xFF4FC3F7);
const _kBg        = Color(0xFF060E22);
const _kCard      = Color(0xFF0D1B3E);
const _kText      = Colors.white;
const _kSubtext   = Color(0xFF8BA0C8);


class LipReadingPage extends StatefulWidget {
  const LipReadingPage({super.key});

  @override
  State<LipReadingPage> createState() => _LipReadingPageState();
}

class _LipReadingPageState extends State<LipReadingPage>
    with WidgetsBindingObserver {

  // ── Camera ──────────────────────────────────────────────────
  CameraController? _camCtrl;
  bool   _camReady   = false;
  bool   _camError   = false;
  String _camErrMsg  = '';

  // ── WebSocket ───────────────────────────────────────────────
  WebSocketChannel?    _channel;
  StreamSubscription?  _wsSub;
  bool _wsConnected = false;
  bool _wsError     = false;

  // ── TTS ─────────────────────────────────────────────────────
  final FlutterTts _tts = FlutterTts();

  // ── Session state ───────────────────────────────────────────
  bool   _modelReady   = false;
  bool   _recording    = false;
  bool   _predicting   = false;
  bool   _faceDetected = false;
  bool   _lipOpen      = false;
  int    _buffered     = 0;
  String _statusMsg    = 'Connecting…';

  double? _bboxX, _bboxY, _bboxW, _bboxH;

  final List<_VsrResult> _results = [];

  // ── Translation state ───────────────────────────────────────
  String? _translatedText;
  bool    _translating     = false;
  String  _translatingLang = '';

  int _lastFrameSentMs = 0;

  // ── Lifecycle ───────────────────────────────────────────────
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
    await _tts.setPitch(1.0);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_camCtrl == null || !_camCtrl!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _stopCamera();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _disconnect();
    _camCtrl?.stopImageStream().catchError((_) {});
    _camCtrl?.dispose();
    _tts.stop();
    super.dispose();
  }

  // ── Camera ──────────────────────────────────────────────────
  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      final cam = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      final ctrl = CameraController(
        cam, ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await ctrl.initialize();
      if (!mounted) return;
      setState(() { _camCtrl = ctrl; _camReady = true; _camError = false; });
      _connectWs();
      _startImageStream();
    } catch (e) {
      if (!mounted) return;
      setState(() { _camError = true; _camErrMsg = e.toString(); _statusMsg = 'Camera error'; });
    }
  }

  void _stopCamera() {
    _camCtrl?.stopImageStream().catchError((_) {});
    _camCtrl?.dispose();
    _camCtrl  = null;
    _camReady = false;
  }

  // ── WebSocket ───────────────────────────────────────────────
  void _connectWs() {
    try {
      _channel = WebSocketChannel.connect(Uri.parse(_kWsUrl));
      _wsSub   = _channel!.stream.listen(_onWsMessage, onError: _onWsError, onDone: _onWsDone);
      setState(() { _wsConnected = true; _wsError = false; _statusMsg = 'Connected — waiting for model…'; });
    } catch (e) {
      setState(() { _wsError = true; _statusMsg = 'Cannot connect. Check Wi-Fi & PC IP.'; });
    }
  }

  void _disconnect() {
    _wsSub?.cancel();
    _channel?.sink.close(ws_status.goingAway);
    _wsConnected = false;
  }

  void _onWsMessage(dynamic raw) {
    if (raw is! String) return;
    final Map<String, dynamic> msg;
    try { msg = jsonDecode(raw) as Map<String, dynamic>; } catch (_) { return; }

    final type = msg['type'] as String? ?? '';
    if (!mounted) return;

    setState(() {
      if (type == 'ready') {
        _modelReady = msg['model'] == true;
        _statusMsg = _modelReady
            ? 'Tap Record to start'
            : 'Model not found: ${msg['error'] ?? ''}';

      } else if (type == 'status') {
        _statusMsg = msg['message'] as String? ?? '';

      } else if (type == 'face') {
        _faceDetected = msg['detected'] == true;
        _lipOpen      = msg['lip_open'] == true;
        _buffered     = (msg['buffered'] as num?)?.toInt() ?? 0;
        final bbox = msg['bbox_norm'] as Map<String, dynamic>?;
        if (bbox != null) {
          const double pad = 2.0;
          final double ox = (bbox['x'] as num).toDouble();
          final double oy = (bbox['y'] as num).toDouble();
          final double ow = (bbox['w'] as num).toDouble();
          final double oh = (bbox['h'] as num).toDouble();
          final double cx = ox + ow / 2;
          final double cy = oy + oh / 2;
          _bboxW = (ow * pad).clamp(0.0, 1.0);
          _bboxH = (oh * pad).clamp(0.0, 1.0);
          _bboxX = (cx - _bboxW! / 2).clamp(0.0, 1.0 - _bboxW!);
          _bboxY = (cy - _bboxH! / 2).clamp(0.0, 1.0 - _bboxH!);
        }

      } else if (type == 'result') {
        _predicting = false;
        final rawTxt   = msg['raw']   as String? ?? '';
        final finalTxt = msg['final'] as String? ?? rawTxt;
        if (finalTxt.isNotEmpty) {
          final actions = (msg['actions'] as List<dynamic>? ?? [])
              .map((a) => Map<String, dynamic>.from(a as Map))
              .toList();
          _results.insert(0, _VsrResult(
            raw:        rawTxt,
            finalText:  finalTxt,
            confidence: (msg['confidence'] as num?)?.toDouble() ?? 0,
            lang:       msg['lang']   as String? ?? '?',
            note:       msg['note']   as String?,
            ts:         TimeOfDay.now(),
            intent:     msg['intent'] as String? ?? '',
            actions:    actions,
          ));
          if (_results.length > 20) _results.removeLast();
          _translatedText  = null; // reset translation on new result
          _translatingLang = '';
          _tts.speak(finalTxt);
        }
        _statusMsg = finalTxt.isEmpty ? 'No speech detected — try again' : 'Done';
        _startImageStream();

      } else if (type == 'error') {
        _predicting = false;
        _recording  = false;
        _statusMsg  = '⚠ ${msg['message'] ?? 'Error'}';
        _startImageStream();
      }
    });
  }

  void _onWsError(Object err) {
    if (!mounted) return;
    setState(() { _wsError = true; _wsConnected = false; _statusMsg = 'Connection lost'; });
  }

  void _onWsDone() {
    if (!mounted) return;
    setState(() { _wsConnected = false; _statusMsg = 'Disconnected'; });
  }

  // ── Frame streaming ─────────────────────────────────────────
  void _startImageStream() {
    if (_camCtrl == null || !_camCtrl!.value.isInitialized) return;
    try {
      _camCtrl!.startImageStream(_onCameraImage);
    } catch (_) {}
  }

  void _onCameraImage(CameraImage image) {
    if (!_wsConnected || _channel == null || _predicting) return;
    final nowMs = DateTime.now().millisecondsSinceEpoch;
    if (nowMs - _lastFrameSentMs < _kFrameIntervalMs) return;
    _lastFrameSentMs = nowMs;
    final Uint8List? jpeg = _extractJpeg(image);
    if (jpeg == null || jpeg.isEmpty) return;
    try { _channel!.sink.add(jpeg); } catch (_) {}
  }

  Uint8List? _extractJpeg(CameraImage image) {
    try { return image.planes[0].bytes; } catch (_) { return null; }
  }

  // ── Controls ────────────────────────────────────────────────
  void _startRecording() {
    if (!_wsConnected || !_modelReady || _recording || _predicting) return;
    _channel!.sink.add('START_RECORDING');
    setState(() { _recording = true; _buffered = 0; _statusMsg = 'Recording — articulate clearly…'; });
  }

  void _stopRecording() {
    if (!_recording) return;
    _channel?.sink.add('STOP_RECORDING');
    setState(() { _recording = false; _statusMsg = 'Tap Result to analyse'; });
  }

  void _predict() {
    if (!_wsConnected || _predicting) return;
    _channel!.sink.add('PREDICT');
    setState(() { _predicting = true; _recording = false; _statusMsg = 'Analysing on CPU… (5-15s)'; });
    _camCtrl?.stopImageStream().catchError((_) {});
  }

  void _reset() {
    _channel?.sink.add('RESET');
    setState(() { _recording = false; _predicting = false; _buffered = 0; _statusMsg = 'Reset — tap Record to start again'; });
  }

  // ── Translation ─────────────────────────────────────────────
  // [specificText] overrides using the full result text (used by action cards
  // that know exactly which phrase to translate, e.g. "translate hello in arabic").
  Future<void> _translateText(String lang, {String? specificText}) async {
    if (_results.isEmpty || _translating) return;
    final textToTranslate = (specificText != null && specificText.isNotEmpty)
        ? specificText
        : _results.first.finalText;
    setState(() { _translating = true; _translatingLang = lang; });
    final t = await ApiService.vsrTranslate(
        text: textToTranslate, targetLanguage: lang);
    if (!mounted) return;
    setState(() {
      _translating     = false;
      _translatingLang = '';
      _translatedText  = t.isNotEmpty ? t : null;
    });
    if (t.isNotEmpty) _tts.speak(t);
  }

  // ── Action cards ─────────────────────────────────────────────
  Future<void> _onActionTap(Map<String, dynamic> action) async {
    final type    = action['type']    as String? ?? '';
    final payload = action['payload'];

    switch (type) {
      case 'copy':
        final text = payload is String ? payload : payload.toString();
        await Clipboard.setData(ClipboardData(text: text));
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Copied to clipboard'),
              duration: Duration(seconds: 1),
              backgroundColor: _kPrimary,
            ),
          );
        }
        break;

      case 'translate':
        // Extract ONLY the specific phrase to translate (not the full sentence).
        String lang         = 'french';
        String? phraseOnly;
        if (payload is Map) {
          phraseOnly = payload['text'] as String?;
          lang       = ((payload['language'] as String?) ?? 'french').toLowerCase();
        } else if (payload is String) {
          lang = payload.toLowerCase();
        }
        await _translateText(lang, specificText: phraseOnly);
        break;

      case 'maps':
        // Opens Google Maps navigation for the extracted destination.
        final dest = Uri.encodeComponent(
            payload is String ? payload : 'near me');
        await _launchUrl('https://maps.google.com/maps?daddr=$dest');
        break;

      case 'glovo':
        // Try Glovo deep-link first; url_launcher falls back to browser.
        final opened = await _tryLaunchUrl('https://glovoapp.com/');
        if (!opened) await _launchUrl('https://glovoapp.com/');
        break;

      case 'web':
        final q = Uri.encodeComponent(payload is String ? payload : '');
        await _launchUrl('https://www.google.com/search?q=$q');
        break;

      case 'youtube':
        final q = Uri.encodeComponent(payload is String ? payload : '');
        await _launchUrl('https://youtube.com/results?search_query=$q');
        break;
    }
  }

  Future<void> _launchUrl(String url) async {
    final uri = Uri.parse(url);
    try {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    } catch (_) {}
  }

  Future<bool> _tryLaunchUrl(String url) async {
    final uri = Uri.parse(url);
    try {
      if (await canLaunchUrl(uri)) {
        await launchUrl(uri, mode: LaunchMode.externalApplication);
        return true;
      }
    } catch (_) {}
    return false;
  }

  // ═══════════════════════════════════════════════════════════
  // UI
  // ═══════════════════════════════════════════════════════════
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _kBg,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(context),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    if (_results.isNotEmpty) ...[
                      _buildLatestResultCard(_results.first),
                      const SizedBox(height: 12),
                      _buildTranslationSection(),
                      if (_results.first.actions.isNotEmpty) ...[
                        const SizedBox(height: 14),
                        _buildActionCards(_results.first.actions),
                      ],
                      const SizedBox(height: 14),
                    ],
                    _buildStatusCard(),
                    const SizedBox(height: 14),
                    _buildCameraCard(),
                    const SizedBox(height: 14),
                    _buildControls(),
                    const SizedBox(height: 14),
                    if (_results.length > 1) ...[
                      const Text('History',
                          style: TextStyle(color: _kText, fontSize: 14, fontWeight: FontWeight.w700)),
                      const SizedBox(height: 8),
                      ..._results.skip(1).map((r) => _ResultCard(result: r, compact: true)),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────
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
            Text('Lip Reading',
                style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w800)),
            Text('Auto-AVSR · English · Real-time',
                style: TextStyle(color: Color(0xFFE7ECFF), fontSize: 12)),
          ]),
        ),
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

  // ── Latest result card ────────────────────────────────────
  Widget _buildLatestResultCard(_VsrResult r) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(colors: [
          _kPrimary.withOpacity(0.08), _kSecondary.withOpacity(0.08),
        ]),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _kPrimary.withOpacity(0.3)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Icon(Icons.format_quote_rounded, color: _kPrimary, size: 18),
          const SizedBox(width: 6),
          const Text('Last transcription',
              style: TextStyle(color: _kPrimary, fontSize: 12,
                  fontWeight: FontWeight.w700, letterSpacing: 0.5)),
          const Spacer(),
          Text('${(r.confidence * 100).toStringAsFixed(0)}%',
              style: TextStyle(
                  color: r.confidence >= 0.6 ? const Color(0xFF00AA55) : Colors.orange,
                  fontSize: 12, fontWeight: FontWeight.w700)),
          const SizedBox(width: 8),
          GestureDetector(
            onTap: () => _tts.speak(r.finalText),
            child: const Icon(Icons.volume_up_rounded, color: _kPrimary, size: 20),
          ),
        ]),
        const SizedBox(height: 10),
        Text(r.finalText,
            style: const TextStyle(color: _kText, fontSize: 22,
                fontWeight: FontWeight.w800, height: 1.3)),
        if (r.raw.isNotEmpty && r.raw != r.finalText) ...[
          const SizedBox(height: 6),
          Text('RAW: ${r.raw}',
              style: const TextStyle(color: _kSubtext, fontSize: 11),
              maxLines: 2, overflow: TextOverflow.ellipsis),
        ],
        if (r.intent.isNotEmpty) ...[
          const SizedBox(height: 8),
          Row(children: [
            const Icon(Icons.lightbulb_outline, color: _kSubtext, size: 13),
            const SizedBox(width: 4),
            Expanded(child: Text(r.intent,
                style: const TextStyle(color: _kSubtext, fontSize: 12, height: 1.4))),
          ]),
        ],
      ]),
    );
  }

  // ── Translation section ───────────────────────────────────
  Widget _buildTranslationSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Padding(
          padding: EdgeInsets.only(bottom: 8),
          child: Text('Translate',
              style: TextStyle(color: _kSubtext, fontSize: 12,
                  fontWeight: FontWeight.w600, letterSpacing: 0.4)),
        ),
        Row(children: [
          Expanded(child: _TranslateBtn(
            flag: '🇫🇷', label: 'French',
            loading: _translating && _translatingLang == 'french',
            onTap:   _translating ? null : () => _translateText('french'),
          )),
          const SizedBox(width: 8),
          Expanded(child: _TranslateBtn(
            flag: '🇸🇦', label: 'Arabic',
            loading: _translating && _translatingLang == 'arabic',
            onTap:   _translating ? null : () => _translateText('arabic'),
          )),
          const SizedBox(width: 8),
          Expanded(child: _TranslateBtn(
            flag: '🇹🇳', label: 'Darija',
            loading: _translating && _translatingLang == 'darija',
            onTap:   _translating ? null : () => _translateText('darija'),
          )),
        ]),
        if (_translatedText != null) ...[
          const SizedBox(height: 10),
          _TranslationCard(
            text: _translatedText!,
            lang: _translatingLang.isEmpty
                ? ''
                : _translatingLang[0].toUpperCase() + _translatingLang.substring(1),
            onSpeak: () => _tts.speak(_translatedText!),
            onCopy: () async {
              await Clipboard.setData(ClipboardData(text: _translatedText!));
              if (mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Translation copied'),
                    duration: Duration(seconds: 1),
                    backgroundColor: _kPrimary,
                  ),
                );
              }
            },
          ),
        ],
      ],
    );
  }

  // ── Action cards ──────────────────────────────────────────
  Widget _buildActionCards(List<Map<String, dynamic>> actions) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Padding(
          padding: EdgeInsets.only(bottom: 8),
          child: Text('Actions',
              style: TextStyle(color: _kSubtext, fontSize: 12,
                  fontWeight: FontWeight.w600, letterSpacing: 0.4)),
        ),
        Wrap(
          spacing: 8, runSpacing: 8,
          children: actions.map((a) => _ActionCard(
            icon:  a['icon']  as String? ?? '•',
            label: a['label'] as String? ?? '',
            type:  a['type']  as String? ?? '',
            onTap: () => _onActionTap(a),
          )).toList(),
        ),
      ],
    );
  }

  // ── Status card ───────────────────────────────────────────
  Widget _buildStatusCard() {
    final Color border = _wsError ? Colors.red.shade300
        : _recording  ? Colors.red
        : _predicting ? const Color(0xFF00AA55)
        : _modelReady ? _kPrimary
        : Colors.orange;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
      decoration: BoxDecoration(
        color: _kCard,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: border, width: 1.5),
      ),
      child: Row(children: [
        if (_recording)
          const _PulsingDot(color: Colors.red)
        else if (_predicting)
          const SizedBox(width: 16, height: 16,
              child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF00AA55)))
        else
          Icon(_modelReady ? Icons.check_circle_outline : Icons.info_outline,
              color: border, size: 18),
        const SizedBox(width: 10),
        Expanded(child: Text(_statusMsg,
            style: const TextStyle(color: _kText, fontSize: 13, fontWeight: FontWeight.w500))),
        if (_buffered > 0)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(
              color: Colors.red.withOpacity(0.1),
              borderRadius: BorderRadius.circular(99),
            ),
            child: Text('$_buffered f',
                style: const TextStyle(color: Colors.red, fontSize: 11, fontWeight: FontWeight.w700)),
          ),
      ]),
    );
  }

  // ── Camera card ───────────────────────────────────────────
  Widget _buildCameraCard() {
    final boxColor = _lipOpen
        ? const Color(0xFF00E887)
        : (_recording ? Colors.red : _kPrimary);

    return Container(
      height: 300,
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: _recording ? Colors.red : _kPrimary.withOpacity(0.3),
          width: _recording ? 2.5 : 1,
        ),
      ),
      clipBehavior: Clip.hardEdge,
      child: Stack(fit: StackFit.expand, children: [
        if (_camReady && _camCtrl != null)
          CameraPreview(_camCtrl!)
        else if (_camError)
          Center(child: Padding(
            padding: const EdgeInsets.all(12),
            child: Text(_camErrMsg,
                style: const TextStyle(color: Colors.red, fontSize: 11),
                textAlign: TextAlign.center)))
        else
          const Center(child: CircularProgressIndicator(color: _kPrimary)),

        if (_faceDetected &&
            _bboxX != null && _bboxY != null && _bboxW != null && _bboxH != null)
          CustomPaint(painter: _MouthBoxPainter(
              bx: _bboxX!, by: _bboxY!, bw: _bboxW!, bh: _bboxH!,
              color: boxColor, lipOpen: _lipOpen)),

        Positioned(top: 8, left: 8, child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(99)),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(_faceDetected ? Icons.face_rounded : Icons.face_outlined,
                color: _faceDetected ? const Color(0xFF00E887) : Colors.grey, size: 14),
            const SizedBox(width: 4),
            Text(_faceDetected ? 'Face' : 'No face',
                style: TextStyle(
                    color: _faceDetected ? const Color(0xFF00E887) : Colors.grey, fontSize: 11)),
          ]),
        )),

        if (_faceDetected)
          Positioned(bottom: 8, left: 0, right: 0, child: Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(99)),
              child: Text(_lipOpen ? '👄 Speaking' : '😐 Silent',
                  style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600)),
            ),
          )),

        if (_recording)
          Positioned(top: 8, right: 8, child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(color: Colors.red, borderRadius: BorderRadius.circular(99)),
            child: const Text('● REC',
                style: TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w700)),
          )),
      ]),
    );
  }

  // ── Controls ──────────────────────────────────────────────
  Widget _buildControls() {
    final canRecord  = _wsConnected && _modelReady && !_recording && !_predicting;
    final canStop    = _recording;
    final canPredict = _wsConnected && !_predicting && !_recording && _buffered > 0;
    final canReset   = _wsConnected && !_predicting;

    return Row(children: [
      Expanded(child: _recording
          ? _CtrlButton(label: 'Stop', icon: Icons.stop_rounded, color: Colors.red,
              onTap: canStop ? _stopRecording : null)
          : _CtrlButton(label: 'Record', icon: Icons.fiber_manual_record_rounded, color: _kPrimary,
              onTap: canRecord ? _startRecording : null)),
      const SizedBox(width: 10),
      Expanded(child: _CtrlButton(
          label: _predicting ? 'Analysing…' : 'Result',
          icon: Icons.psychology_rounded,
          color: const Color(0xFF00AA55),
          onTap: canPredict ? _predict : null)),
      const SizedBox(width: 10),
      SizedBox(width: 48, child: _CtrlButton(
          label: '', icon: Icons.refresh_rounded, color: _kSubtext, compact: true,
          onTap: canReset ? _reset : null)),
    ]);
  }
}


// ═══════════════════════════════════════════════════════════════
// Sub-widgets
// ═══════════════════════════════════════════════════════════════

class _TranslateBtn extends StatelessWidget {
  final String      flag;
  final String      label;
  final bool        loading;
  final VoidCallback? onTap;
  const _TranslateBtn({required this.flag, required this.label,
      required this.loading, this.onTap});

  @override
  Widget build(BuildContext context) {
    final enabled = onTap != null && !loading;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 42,
        decoration: BoxDecoration(
          color: enabled ? _kCard : _kCard.withOpacity(0.5),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: enabled ? _kPrimary.withOpacity(0.5) : _kSubtext.withOpacity(0.2),
          ),
        ),
        child: loading
            ? const Center(child: SizedBox(width: 16, height: 16,
                child: CircularProgressIndicator(strokeWidth: 2, color: _kPrimary)))
            : Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                Text(flag, style: const TextStyle(fontSize: 15)),
                const SizedBox(width: 5),
                Text(label,
                    style: TextStyle(
                        color: enabled ? _kText : _kSubtext,
                        fontSize: 12, fontWeight: FontWeight.w600)),
              ]),
      ),
    );
  }
}


class _TranslationCard extends StatelessWidget {
  final String       text;
  final String       lang;
  final VoidCallback onSpeak;
  final VoidCallback onCopy;
  const _TranslationCard({required this.text, required this.lang,
      required this.onSpeak, required this.onCopy});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _kCard,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _kSecondary.withOpacity(0.35)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          const Icon(Icons.translate_rounded, color: _kSecondary, size: 14),
          const SizedBox(width: 6),
          Text(lang.isNotEmpty ? lang : 'Translation',
              style: const TextStyle(color: _kSecondary, fontSize: 11,
                  fontWeight: FontWeight.w700, letterSpacing: 0.4)),
          const Spacer(),
          GestureDetector(onTap: onSpeak,
              child: const Icon(Icons.volume_up_rounded, color: _kSecondary, size: 18)),
          const SizedBox(width: 10),
          GestureDetector(onTap: onCopy,
              child: const Icon(Icons.copy_rounded, color: _kSubtext, size: 16)),
        ]),
        const SizedBox(height: 8),
        Text(text,
            style: const TextStyle(color: _kText, fontSize: 17,
                fontWeight: FontWeight.w600, height: 1.4)),
      ]),
    );
  }
}


class _ActionCard extends StatelessWidget {
  final String       icon;
  final String       label;
  final String       type;
  final VoidCallback onTap;
  const _ActionCard({required this.icon, required this.label,
      required this.type, required this.onTap});

  Color get _color {
    switch (type) {
      case 'maps':      return const Color(0xFF00AA55);
      case 'glovo':     return const Color(0xFFFF6B35);
      case 'translate': return const Color(0xFF4C6FFF);
      case 'web':       return const Color(0xFF4FC3F7);
      case 'youtube':   return Colors.red;
      default:          return _kSubtext;
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: _color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: _color.withOpacity(0.4)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Text(icon, style: const TextStyle(fontSize: 16)),
          const SizedBox(width: 6),
          Text(label,
              style: TextStyle(color: _color, fontSize: 13, fontWeight: FontWeight.w600)),
        ]),
      ),
    );
  }
}


class _CtrlButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final Color color;
  final VoidCallback? onTap;
  final bool compact;
  const _CtrlButton({required this.label, required this.icon, required this.color,
      this.onTap, this.compact = false});

  @override
  Widget build(BuildContext context) {
    final enabled = onTap != null;
    return Material(
      color: enabled ? color : color.withOpacity(0.35),
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Container(
          height: 52,
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            Icon(icon, color: Colors.white, size: 20),
            if (!compact && label.isNotEmpty) ...[
              const SizedBox(width: 6),
              Flexible(child: Text(label,
                  style: const TextStyle(color: Colors.white,
                      fontSize: 14, fontWeight: FontWeight.w700),
                  overflow: TextOverflow.ellipsis)),
            ],
          ]),
        ),
      ),
    );
  }
}


class _ResultCard extends StatelessWidget {
  final _VsrResult result;
  final bool compact;
  const _ResultCard({required this.result, this.compact = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: _kCard,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _kPrimary.withOpacity(0.15)),
      ),
      child: Row(children: [
        Expanded(child: Text(result.finalText,
            style: const TextStyle(color: _kText, fontSize: 14, fontWeight: FontWeight.w700))),
        const SizedBox(width: 8),
        Text('${(result.confidence * 100).toStringAsFixed(0)}%',
            style: TextStyle(
                color: result.confidence >= 0.6 ? const Color(0xFF00AA55) : Colors.orange,
                fontSize: 11, fontWeight: FontWeight.w700)),
        const SizedBox(width: 6),
        Text('${result.ts.hour.toString().padLeft(2,'0')}:'
             '${result.ts.minute.toString().padLeft(2,'0')}',
            style: const TextStyle(color: _kSubtext, fontSize: 10)),
      ]),
    );
  }
}


class _PulsingDot extends StatefulWidget {
  final Color color;
  const _PulsingDot({required this.color});
  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}

class _PulsingDotState extends State<_PulsingDot>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 800))
      ..repeat(reverse: true);
    _anim = Tween<double>(begin: 0.4, end: 1.0).animate(_ctrl);
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) =>
      FadeTransition(opacity: _anim, child: Icon(Icons.circle, color: widget.color, size: 14));
}


// ── Mouth bounding box painter ────────────────────────────────
class _MouthBoxPainter extends CustomPainter {
  final double bx, by, bw, bh;
  final Color color;
  final bool lipOpen;
  const _MouthBoxPainter({required this.bx, required this.by,
      required this.bw, required this.bh, required this.color, required this.lipOpen});

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Rect.fromLTWH(
        bx * size.width, by * size.height, bw * size.width, bh * size.height);

    canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(6)),
        Paint()..color = color.withOpacity(0.25)..style = PaintingStyle.stroke..strokeWidth = 8);

    canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(6)),
        Paint()..color = color..style = PaintingStyle.stroke..strokeWidth = 2.5);

    final ap = Paint()..color = color..style = PaintingStyle.stroke
        ..strokeWidth = 3.5..strokeCap = StrokeCap.round;
    const cl = 12.0;
    final l = rect.left; final t = rect.top; final r = rect.right; final b = rect.bottom;
    canvas.drawLine(Offset(l, t + cl), Offset(l, t), ap);
    canvas.drawLine(Offset(l, t), Offset(l + cl, t), ap);
    canvas.drawLine(Offset(r - cl, t), Offset(r, t), ap);
    canvas.drawLine(Offset(r, t), Offset(r, t + cl), ap);
    canvas.drawLine(Offset(l, b - cl), Offset(l, b), ap);
    canvas.drawLine(Offset(l, b), Offset(l + cl, b), ap);
    canvas.drawLine(Offset(r - cl, b), Offset(r, b), ap);
    canvas.drawLine(Offset(r, b), Offset(r, b - cl), ap);
  }

  @override
  bool shouldRepaint(_MouthBoxPainter old) =>
      old.bx != bx || old.by != by || old.bw != bw ||
      old.bh != bh || old.color != color || old.lipOpen != lipOpen;
}


// ── Data ──────────────────────────────────────────────────────
class _VsrResult {
  final String raw;
  final String finalText;
  final double confidence;
  final String lang;
  final String? note;
  final TimeOfDay ts;
  final String intent;
  final List<Map<String, dynamic>> actions;
  const _VsrResult({
    required this.raw,
    required this.finalText,
    required this.confidence,
    required this.lang,
    required this.note,
    required this.ts,
    required this.intent,
    required this.actions,
  });
}
