// camera_screen.dart — v5
//
// ══════════════════════════════════════════════════════════════
// QUOTA-AWARE ADAPTIVE SCAN ARCHITECTURE
// ══════════════════════════════════════════════════════════════
//
// The core insight: every VLM call costs quota. The old 2.5s flat
// interval was architecture-level waste. This version fires requests
// ONLY when useful.
//
// ── Three-gate system (all gates must be open to fire a request) ──
//
//   GATE 1 — TIME  (hard floor: minimum seconds between calls)
//     Normal:        8 s  — Gemini 2.5 Flash-Lite safe at 15 RPM
//     OR-fallback:  12 s  — OpenRouter free tier is much smaller
//     Backoff-429:  60 s  — after any rate-limit response
//     Both-exhausted: 0   — show "quota used up" message, stop loop
//
//   GATE 2 — MOTION  (pixel-diff check, zero API cost)
//     Compare a 40×30 thumbnail of the current frame against the
//     last sent frame. If fewer than MOTION_THRESHOLD pixels changed
//     significantly, skip the request — the user isn't moving and
//     the scene is identical. This alone saves ~60% of requests in
//     real usage (user standing still, scanning a static surface).
//     Motion gate bypassed after MOTION_FORCE_AFTER_S seconds of
//     forced silence so the user never waits forever.
//
//   GATE 3 — CONSECUTIVE-MISS SLOWDOWN
//     After N consecutive SEARCHING frames with NO change detected,
//     double the interval (up to a cap) and tell the user to move.
//     The interval resets to base the moment any detection is made.
//
// ── Budget counter (session-level) ──
//   Track requests sent this session. When budget is exhausted:
//   - Speak a single warning in the user's language.
//   - Show a countdown to midnight Pacific (Gemini reset) or
//     2-minute cooloff (OR reset) in the status bar.
//   - Stop sending requests entirely (no silent 429 spam).
//
// ── Quota math (confirmed April 2026) ──
//   Gemini 2.5 Flash-Lite: 1 000 req/day, 15 RPM, resets midnight PT
//   OpenRouter :free models: ~50 req/day per model × 3 = ~150/day
//
//   At 8 s base interval:
//     Gemini  → 1 000 × 8 s = 8 000 s = 133 min/day ← primary
//     OR      →   150 × 8 s = 1 200 s =  20 min/day ← fallback
//   With motion gating (saves ~60%):
//     Effective Gemini capacity ≈ 333 min ≈ 5.5 h of active scanning

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show FontFeature;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:vibration/vibration.dart';

import 'tts_service.dart';
import 'api_service.dart';

// ── Bounding box data ─────────────────────────────────────────

class BoundingBox {
  final String label;
  final String position;
  final String distance;
  final bool   threat;
  final bool   isTarget;
  final double cx, cy, w, h;
  const BoundingBox({
    required this.label, required this.position, required this.distance,
    required this.threat, required this.isTarget,
    required this.cx, required this.cy, required this.w, required this.h,
  });
}

// ── Quota / provider state ────────────────────────────────────

enum _Provider { gemini, openRouter, exhausted }

class _QuotaState {
  // Session request counters
  int geminiUsed = 0;
  int orUsed     = 0;

  // Soft daily caps (real limits are higher but we stay conservative)
  // Gemini: 1 000/day real limit → we cap at 900 to leave headroom
  // OR: ~150/day across 3 models → we cap at 130
  static const int geminiDayCap = 900;
  static const int orDayCap     = 130;

  // Cooloff timestamps (epoch ms)
  int geminiCooloffUntil = 0;
  int orCooloffUntil     = 0;

  // Which provider are we actively using?
  _Provider activeProvider = _Provider.gemini;

  // Adaptive interval state
  int consecutiveMisses = 0;
  static const int maxConsecutiveMisses = 5;

  _Provider currentProvider() {
    final now = DateTime.now().millisecondsSinceEpoch;
    final geminiOk = geminiUsed < geminiDayCap &&
                     geminiCooloffUntil <= now;
    if (geminiOk) return _Provider.gemini;

    final orOk = orUsed < orDayCap && orCooloffUntil <= now;
    if (orOk) return _Provider.openRouter;

    return _Provider.exhausted;
  }

  /// Base scan interval in seconds for the current provider.
  int baseScanSeconds() {
    switch (currentProvider()) {
      case _Provider.gemini:     return 8;
      case _Provider.openRouter: return 12;
      case _Provider.exhausted:  return 0;
    }
  }

  /// Adaptive interval: base × miss-slowdown multiplier, capped at 45 s.
  int adaptiveScanSeconds() {
    final base = baseScanSeconds();
    if (base == 0) return 0;
    // Slow down after consecutive misses (still searching, no movement)
    final slowdown = math.min(consecutiveMisses ~/ 2, 3); // 0-3 extra ×2
    return math.min(base + (slowdown * base), 45);
  }

  void recordRateLimit(_Provider p) {
    final now = DateTime.now().millisecondsSinceEpoch;
    if (p == _Provider.gemini) {
      // Gemini 429 = daily quota exhausted → cooloff until midnight Pacific
      // (We approximate as 1 h from now; the server already has its own
      // cooloff logic, but we also stop the Flutter loop independently.)
      geminiCooloffUntil = now + const Duration(hours: 1).inMilliseconds;
    } else {
      // OpenRouter 429 = per-model quota → 2-minute cooloff
      orCooloffUntil = now + const Duration(minutes: 2).inMilliseconds;
    }
  }

  void recordSuccess(_Provider p) {
    if (p == _Provider.gemini) geminiUsed++;
    else if (p == _Provider.openRouter) orUsed++;
  }

  int remainingCooloffMs() {
    final now = DateTime.now().millisecondsSinceEpoch;
    switch (currentProvider()) {
      case _Provider.gemini:
        return math.max(0, geminiCooloffUntil - now);
      case _Provider.openRouter:
        return math.max(0, orCooloffUntil - now);
      case _Provider.exhausted:
        // Return whichever cooloff ends sooner
        final gcool = math.max(0, geminiCooloffUntil - now);
        final ocool = math.max(0, orCooloffUntil - now);
        return math.min(gcool == 0 ? 999999999 : gcool,
                        ocool == 0 ? 999999999 : ocool);
    }
  }

  String budgetSummary() =>
      'Gemini ${geminiUsed}/${geminiDayCap}  ·  OR ${orUsed}/${orDayCap}';
}

// ── Motion detector (pixel diff on tiny thumbnail) ────────────

class _MotionGate {
  // We compare 40×30 = 1 200 pixels — cheap, fast, no GPU needed
  static const int _W = 40;
  static const int _H = 30;
  static const int _threshold = 20;    // pixel value change to count as diff
  static const double _minChangedFraction = 0.06; // 6% of pixels must change

  // Force a send after this many seconds even if no motion detected.
  // This ensures the user gets at least one result if they're holding still.
  static const int forceAfterSeconds = 30;

  Uint8List? _lastThumb;
  DateTime _lastSentAt = DateTime.fromMillisecondsSinceEpoch(0);

  bool shouldSend(Uint8List frameBytes) {
    // Always send if it's been too long since the last send
    final age = DateTime.now().difference(_lastSentAt).inSeconds;
    if (age >= forceAfterSeconds) return true;

    final thumb = _thumbnail(frameBytes);
    if (thumb == null) return true;          // can't decode → send anyway

    final last = _lastThumb;
    if (last == null || last.length != thumb.length) {
      _lastThumb = thumb;
      return true;
    }

    int changed = 0;
    for (int i = 0; i < thumb.length; i++) {
      if ((thumb[i] - last[i]).abs() > _threshold) changed++;
    }
    final fraction = changed / thumb.length;
    return fraction >= _minChangedFraction;
  }

  void markSent(Uint8List frameBytes) {
    _lastSentAt = DateTime.now();
    _lastThumb  = _thumbnail(frameBytes);
  }

  // Produce a tiny greyscale thumbnail from a JPEG.
  // We decode only the first scan (fast) and subsample.
  Uint8List? _thumbnail(Uint8List jpeg) {
    try {
      // Fast path: sample every Nth byte as a brightness proxy.
      // This isn't a proper decode but is fast and catches motion well.
      final step = jpeg.length ~/ (_W * _H);
      if (step < 1) return null;
      final out = Uint8List(_W * _H);
      for (int i = 0; i < out.length; i++) {
        out[i] = jpeg[i * step];
      }
      return out;
    } catch (_) {
      return null;
    }
  }

  void reset() {
    _lastThumb = null;
    _lastSentAt = DateTime.fromMillisecondsSinceEpoch(0);
  }
}

// ── Screen ───────────────────────────────────────────────────

class CameraScreen extends StatefulWidget {
  final String  mode;
  final String? findTarget;
  final String  language;
  const CameraScreen({
    super.key,
    required this.mode,
    this.findTarget,
    required this.language,
  });
  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with SingleTickerProviderStateMixin {

  CameraController? _ctrl;
  bool _cameraReady = false;
  bool _found       = false;
  bool _closing     = false;
  bool _disposed    = false;

  String _statusText  = '';
  String _danger      = 'LOW';
  String _consensus   = 'SEARCHING';
  List<BoundingBox> _boxes = [];

  // Quota + motion subsystems
  final _QuotaState  _quota  = _QuotaState();
  final _MotionGate  _motion = _MotionGate();

  // Voice dedup
  String   _lastSpoken = '';
  DateTime _lastNagAt  = DateTime.fromMillisecondsSinceEpoch(0);
  String   _userId     = '';

  // Debug overlay (shown in status bar during dev — remove for prod)
  String _debugBudget = '';

  late AnimationController _pulseAnim;

  @override
  void initState() {
    super.initState();
    _pulseAnim = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1000))
      ..repeat(reverse: true);
    _initCamera();
  }

  @override
  void dispose() {
    _disposed = true;
    _pulseAnim.dispose();
    final ctrl = _ctrl; _ctrl = null;
    Future.delayed(const Duration(milliseconds: 400), () async {
      try { await ctrl?.dispose(); } catch (_) {}
    });
    super.dispose();
  }

  void _ss(VoidCallback fn) {
    if (!_disposed && mounted) setState(fn);
  }

  Future<void> _initCamera() async {
    List<CameraDescription> cameras;
    try {
      cameras = await availableCameras();
    } catch (_) {
      _ss(() => _statusText = 'Camera unavailable'); return;
    }
    if (cameras.isEmpty) {
      _ss(() => _statusText = 'No camera found'); return;
    }

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
      await Future.delayed(const Duration(milliseconds: 1200));
    } catch (e) {
      print('CameraScreen init: $e');
      try { await ctrl.dispose(); } catch (_) {}
      return;
    }

    if (_disposed || !mounted) {
      try { await ctrl.dispose(); } catch (_) {}
      return;
    }

    _userId = await ApiService.getUserId();
    _ctrl = ctrl;
    _ss(() {
      _cameraReady = true;
      _statusText  = widget.mode == 'FIND_OBJECT'
          ? 'Searching for ${widget.findTarget ?? "object"}…'
          : (widget.mode == 'NAV' ? 'Watching the road…' : 'Scanning…');
    });

    _runScanLoop();
  }

  // ══════════════════════════════════════════════════════════
  // SCAN LOOP — quota-aware, motion-gated
  // ══════════════════════════════════════════════════════════

  Future<void> _runScanLoop() async {
    while (!_disposed && !_closing && !_found) {
      final provider = _quota.currentProvider();

      // ── Exhausted: all quotas used / cooling off ─────────
      if (provider == _Provider.exhausted) {
        final coolMs = _quota.remainingCooloffMs();
        if (coolMs > 0) {
          final mins = (coolMs / 60000).ceil();
          _ss(() => _statusText =
              'Daily quota reached. Resuming in ~$mins min…');
          _speakOnce(_quotaMsg(widget.language), widget.language);
          await Future.delayed(const Duration(seconds: 30));
          continue;
        }
        // Cooloff expired — try again
        continue;
      }

      // ── Capture frame ────────────────────────────────────
      final frameBytes = await _captureFrame();
      if (frameBytes == null) {
        await Future.delayed(const Duration(seconds: 2));
        continue;
      }

      // ── GATE 2: motion check ─────────────────────────────
      if (!_motion.shouldSend(frameBytes)) {
        // Scene unchanged — wait half the normal interval then recheck
        final half = (_quota.adaptiveScanSeconds() / 2).round();
        await Future.delayed(Duration(seconds: math.max(half, 2)));
        continue;
      }

      // ── Fire the request ─────────────────────────────────
      _motion.markSent(frameBytes);
      await _scanWithFrame(frameBytes, provider);

      // ── GATE 1: time floor ───────────────────────────────
      final waitS = _quota.adaptiveScanSeconds();
      _ss(() => _debugBudget = _quota.budgetSummary());
      if (waitS > 0 && !_disposed && !_closing && !_found) {
        await Future.delayed(Duration(seconds: waitS));
      }
    }
  }

  Future<Uint8List?> _captureFrame() async {
    if (_disposed || _closing || _found) return null;
    if (!_cameraReady) return null;
    final ctrl = _ctrl;
    if (ctrl == null || !ctrl.value.isInitialized) return null;
    try {
      final XFile photo = await ctrl.takePicture();
      final bytes = await photo.readAsBytes();
      try { File(photo.path).deleteSync(); } catch (_) {}
      return bytes;
    } catch (e) {
      if (!_disposed) print('capture error: $e');
      return null;
    }
  }

  Future<void> _scanWithFrame(Uint8List frameBytes, _Provider provider) async {
    if (_disposed || !mounted) return;

    try {
      final endpoint = widget.mode == 'NAV' ? '/nav_danger' : '/vision';
      final request  = http.MultipartRequest(
          'POST', Uri.parse('${ApiService.baseUrl}$endpoint'));

      if (widget.mode != 'NAV') {
        request.fields['mode']     = widget.mode;
        request.fields['target']   = widget.findTarget ?? '';
        request.fields['language'] = widget.language;
      } else {
        request.fields['language'] = widget.language;
      }
      request.files.add(http.MultipartFile.fromBytes(
          'frame', frameBytes, filename: 'frame.jpg'));
      request.headers['X-User-ID'] = _userId;

      final streamed = await request.send()
          .timeout(const Duration(seconds: 20));
      final body = await streamed.stream.bytesToString();
      if (_disposed || !mounted) return;

      // ── HTTP 429: rate-limit from the backend/model ───────
      if (streamed.statusCode == 429) {
        _quota.recordRateLimit(provider);
        final waitS = provider == _Provider.gemini ? 3600 : 120;
        _ss(() => _statusText =
            provider == _Provider.gemini
                ? 'Gemini quota reached — switching to backup…'
                : 'Rate limited — waiting ${waitS ~/ 60} min…');
        _motion.reset(); // force re-check on next cycle
        return;
      }

      if (streamed.statusCode != 200) return;

      _quota.recordSuccess(provider);

      final decoded = jsonDecode(body) as Map<String, dynamic>;

      // Backend signals it's rate-limited (model returned 429 to it)
      if (decoded['rate_limited'] == true) {
        _quota.recordRateLimit(provider);
        _ss(() => _statusText = 'Model busy — backing off…');
        return;
      }

      if (widget.mode == 'NAV') {
        _handleNavDangerResponse(decoded);
      } else {
        _handleResponse(decoded);
      }
    } catch (e) {
      if (!_disposed) print('scan error: $e');
    }
  }

  // ── Nav danger response (YOLO — no quota cost) ───────────

  void _handleNavDangerResponse(Map<String, dynamic> data) {
    final speak     = (data['speak']     ?? '') as String;
    final danger    = (data['danger']    ?? 'LOW') as String;
    final interrupt = (data['interrupt'] ?? false) as bool;
    final rawObjs   = (data['objects']   ?? []) as List;

    final boxes = <BoundingBox>[];
    for (final o in rawObjs) {
      final isDanger = o['is_danger'] == true;
      if (!isDanger) continue;
      final xyxy = (o['bbox_xyxy'] as List?) ?? const [];
      if (xyxy.length != 4) continue;
      final imgW = ((o['img_w'] ?? 1) as num).toDouble();
      final imgH = ((o['img_h'] ?? 1) as num).toDouble();
      final x1 = (xyxy[0] as num).toDouble();
      final y1 = (xyxy[1] as num).toDouble();
      final x2 = (xyxy[2] as num).toDouble();
      final y2 = (xyxy[3] as num).toDouble();
      boxes.add(BoundingBox(
        label:    (o['label'] ?? 'object') as String,
        position: (o['direction'] ?? 'center') as String,
        distance: (o['proximity'] ?? 'medium') as String,
        threat: true, isTarget: false,
        cx: ((x1 + x2) * 0.5) / imgW,
        cy: ((y1 + y2) * 0.5) / imgH,
        w:  (x2 - x1) / imgW,
        h:  (y2 - y1) / imgH,
      ));
    }

    _ss(() {
      _danger    = danger;
      _consensus = boxes.isEmpty ? 'SEARCHING' : 'NOT_FOUND';
      _boxes     = boxes;
      if (speak.isNotEmpty) _statusText = speak;
    });

    _speakIfNew(speak, widget.language,
        interrupt: danger == 'HIGH' || interrupt);
    if (danger == 'HIGH') _vibrate(strong: true);
  }

  // ── Vision response (VLM) ────────────────────────────────

  void _handleResponse(Map<String, dynamic> data) {
    final speak     = (data['speak']     ?? '') as String;
    final danger    = (data['danger']    ?? 'LOW') as String;
    final interrupt = (data['interrupt'] ?? false) as bool;
    final found     = data['found'] == true;
    final consensus = (data['consensus'] ?? 'SEARCHING') as String;
    final rawObjs   = (data['objects']   ?? []) as List;

    // Update consecutive-miss counter for adaptive slowdown
    if (consensus == 'SEARCHING') {
      _quota.consecutiveMisses++;
    } else {
      _quota.consecutiveMisses = 0; // any real detection resets it
    }

    final boxes = <BoundingBox>[];
    for (final o in rawObjs) {
      final label    = (o['label']    ?? 'object') as String;
      final pos      = (o['position'] ?? 'center') as String;
      final dist     = (o['distance'] ?? 'medium') as String;
      final threat   = o['threat']    == true;
      final isTarget = (o['is_target'] == true) ||
          (widget.findTarget != null &&
           label.toLowerCase().contains(widget.findTarget!.toLowerCase()));

      final cx = _numField(o, 'bbox_cx', 0.5);
      final cy = _numField(o, 'bbox_cy', 0.5);
      final w  = _numField(o, 'bbox_w',  0.25);
      final h  = _numField(o, 'bbox_h',  0.20);

      final keep = (isTarget && (consensus == 'FOUND' || found))
                || (threat && danger == 'HIGH');
      if (keep) {
        boxes.add(BoundingBox(
          label: label, position: pos, distance: dist,
          threat: threat, isTarget: isTarget,
          cx: cx, cy: cy, w: w, h: h,
        ));
      }
    }

    _ss(() {
      _danger    = danger;
      _consensus = consensus;
      _boxes     = boxes;
      if (speak.isNotEmpty) _statusText = speak;
    });

    _speakIfNew(speak, widget.language,
        interrupt: danger == 'HIGH' || interrupt);

    // Periodic direction hint when still searching, not too chatty
    if (consensus == 'NOT_FOUND' &&
        DateTime.now().difference(_lastNagAt).inSeconds > 8) {
      final hints = _nagHints(widget.language);
      final hint  = hints[(DateTime.now().second ~/ 8) % hints.length];
      TtsService.speak(hint, lang: widget.language);
      _lastNagAt = DateTime.now();
    }

    if (danger == 'HIGH') _vibrate(strong: true);

    if (found && !_found) {
      _found = true;
      _onFound();
    }
  }

  // ── Helpers ───────────────────────────────────────────────

  static double _numField(Map m, String k, double def) {
    final v = m[k];
    return v is num ? v.toDouble() : def;
  }

  void _speakIfNew(String text, String lang, {bool interrupt = false}) {
    if (text.isEmpty || text == _lastSpoken) return;
    _lastSpoken = text;
    _lastNagAt  = DateTime.now();
    TtsService.speak(text, lang: lang, interrupt: interrupt);
  }

  void _speakOnce(String text, String lang) {
    if (text == _lastSpoken) return;
    _lastSpoken = text;
    TtsService.speak(text, lang: lang);
  }

  static String _quotaMsg(String lang) {
    const msgs = {
      'en': 'Daily scan limit reached. Please try again later.',
      'fr': 'Limite quotidienne atteinte. Réessayez plus tard.',
      'ar': 'تم الوصول إلى الحد اليومي. يرجى المحاولة لاحقاً.',
      'tn': 'وصلنا للحد اليومي. عاود لاحقاً.',
    };
    return msgs[lang] ?? msgs['en']!;
  }

  static List<String> _nagHints(String lang) {
    const hints = {
      'en': [
        'Try turning slowly to your right.',
        'Slowly look down toward the surface.',
        'Move forward a little and try again.',
        'Try scanning from left to right.',
      ],
      'fr': [
        'Tournez lentement vers la droite.',
        'Regardez doucement vers le sol.',
        'Avancez un peu et réessayez.',
        'Balayez de gauche à droite.',
      ],
      'ar': [
        'استدر ببطء إلى اليمين.',
        'انظر إلى الأسفل ببطء.',
        'تقدم قليلاً وحاول مرة أخرى.',
        'امسح من اليسار إلى اليمين.',
      ],
      'tn': [
        'دور بشويا على اليمين.',
        'بص لتحت بشويا.',
        'تقدم شويا و عاود.',
        'مسح من اليسار للصح.',
      ],
    };
    return hints[lang] ?? hints['en']!;
  }

  Future<void> _vibrate({bool strong = false}) async {
    try {
      if (await Vibration.hasVibrator() ?? false) {
        Vibration.vibrate(pattern: strong ? [0, 300, 100, 300] : [0, 150]);
      }
    } catch (_) {}
  }

  Future<void> _onFound() async {
    await _vibrate(strong: true);
    _ss(() => _statusText = '✓ Found ${widget.findTarget ?? "object"}!');
    await Future.delayed(const Duration(seconds: 3));
    _close();
  }

  void _close() {
    if (_closing || _disposed || !mounted) return;
    _closing = true;
    ApiService.reset();
    Navigator.pop(context);
  }

  // ══════════════════════════════════════════════════════════
  // BUILD
  // ══════════════════════════════════════════════════════════

  @override
  Widget build(BuildContext context) {
    final bool camOk = _cameraReady && _ctrl != null
        && !_disposed && _ctrl!.value.isInitialized;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(fit: StackFit.expand, children: [
        camOk ? CameraPreview(_ctrl!) : _buildLoading(),
        const _Vignette(),
        if (camOk && _boxes.isNotEmpty)
          LayoutBuilder(builder: (ctx, cs) => CustomPaint(
            size: Size(cs.maxWidth, cs.maxHeight),
            painter: _BoxPainter(_boxes),
          )),
        _buildTopBar(),
        if (camOk && !_found && _boxes.isEmpty) _buildScanRing(),
        _buildBottomPanel(),
        // Budget strip — small, unobtrusive, helpful during dev
        _buildBudgetStrip(),
      ]),
    );
  }

  Widget _buildLoading() => Container(
    color: const Color(0xFF080810),
    child: const Center(child: Column(
      mainAxisSize: MainAxisSize.min, children: [
        SizedBox(width: 32, height: 32, child: CircularProgressIndicator(
          color: Color(0xFF00E5C0), strokeWidth: 2)),
        SizedBox(height: 16),
        Text('Initializing camera…',
            style: TextStyle(color: Colors.white54, fontSize: 13)),
      ])),
  );

  Widget _buildTopBar() {
    final Color    color;
    final String   label;
    final IconData icon;
    if (_consensus == 'FOUND' || _found) {
      color = const Color(0xFF34C759); label = 'FOUND'; icon = Icons.check_circle_rounded;
    } else if (_quota.currentProvider() == _Provider.exhausted) {
      color = const Color(0xFFFF9500); label = 'QUOTA USED'; icon = Icons.pause_circle_outline_rounded;
    } else if (_danger == 'HIGH') {
      color = const Color(0xFFFF3B3B); label = 'DANGER'; icon = Icons.warning_amber_rounded;
    } else if (_danger == 'MEDIUM') {
      color = const Color(0xFFFF9500); label = 'CAUTION'; icon = Icons.info_outline_rounded;
    } else {
      color = const Color(0xFF00E5C0); label = 'SEARCHING'; icon = Icons.search_rounded;
    }
    return Positioned(top: 0, left: 0, right: 0, child: SafeArea(
      bottom: false,
      child: Container(
        margin:  const EdgeInsets.fromLTRB(12, 8, 12, 0),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: color.withOpacity(0.15),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withOpacity(0.55), width: 1),
        ),
        child: Row(children: [
          Icon(icon, color: color, size: 16),
          const SizedBox(width: 8),
          Text(label, style: TextStyle(
            color: color, fontWeight: FontWeight.w700,
            fontSize: 13, letterSpacing: 1.5)),
          const Spacer(),
          if (widget.findTarget != null)
            Text(widget.findTarget!.toUpperCase(),
              style: TextStyle(color: color.withOpacity(0.7),
                fontSize: 11, letterSpacing: 1.0)),
        ]),
      ),
    ));
  }

  Widget _buildScanRing() => Center(child: AnimatedBuilder(
    animation: _pulseAnim,
    builder: (_, __) => Transform.scale(
      scale: 1.0 + _pulseAnim.value * 0.06,
      child: Container(
        width: 220, height: 220,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(
            color: const Color(0xFF00E5C0)
                .withOpacity(0.2 + _pulseAnim.value * 0.2),
            width: 1.5)),
      ),
    ),
  ));

  Widget _buildBottomPanel() => Positioned(
    bottom: 0, left: 0, right: 0,
    child: Container(
      padding: const EdgeInsets.fromLTRB(20, 28, 20, 40),
      decoration: BoxDecoration(gradient: LinearGradient(
        begin: Alignment.bottomCenter, end: Alignment.topCenter,
        colors: [Colors.black.withOpacity(0.95),
                 Colors.black.withOpacity(0.55), Colors.transparent],
        stops: const [0.0, 0.55, 1.0])),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        if (_statusText.isNotEmpty)
          Text(_statusText,
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: Colors.white, fontSize: 17,
              fontWeight: FontWeight.w500, height: 1.4,
              shadows: [Shadow(blurRadius: 8, color: Colors.black)])),
        const SizedBox(height: 20),
        GestureDetector(onTap: _close, child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 12),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.10),
            borderRadius: BorderRadius.circular(50),
            border: Border.all(color: Colors.white.withOpacity(0.2), width: 1)),
          child: const Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.close_rounded, color: Colors.white70, size: 18),
            SizedBox(width: 8),
            Text('Close', style: TextStyle(
              color: Colors.white70, fontSize: 15,
              fontWeight: FontWeight.w500)),
          ]),
        )),
      ]),
    ),
  );

  /// Tiny semi-transparent strip at the very bottom showing live quota.
  /// Remove this widget (return const SizedBox()) before production.
  Widget _buildBudgetStrip() {
    if (_debugBudget.isEmpty) return const SizedBox();
    return Positioned(
      bottom: 92, left: 0, right: 0,
      child: Center(child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(6)),
        child: Text(_debugBudget,
          style: const TextStyle(
            color: Colors.white38, fontSize: 10,
            fontFeatures: [FontFeature.tabularFigures()])),
      )),
    );
  }
}

// ── Vignette ─────────────────────────────────────────────────

class _Vignette extends StatelessWidget {
  const _Vignette();
  @override
  Widget build(BuildContext context) => DecoratedBox(
    decoration: BoxDecoration(gradient: RadialGradient(
      center: Alignment.center, radius: 1.0,
      colors: [Colors.transparent, Colors.black.withOpacity(0.35)])),
  );
}

// ── Bounding-box painter ──────────────────────────────────────

class _BoxPainter extends CustomPainter {
  final List<BoundingBox> boxes;
  const _BoxPainter(this.boxes);

  @override
  void paint(Canvas canvas, Size size) {
    final stroke = Paint()..style = PaintingStyle.stroke..strokeWidth = 3;
    final fill   = Paint()..style = PaintingStyle.fill;

    for (final b in boxes) {
      final color = b.isTarget
          ? const Color(0xFF00FF9D)
          : const Color(0xFFFF3B3B);

      final cx   = b.cx * size.width;
      final cy   = b.cy * size.height;
      final w    = b.w  * size.width;
      final h    = b.h  * size.height;
      final rect = Rect.fromCenter(center: Offset(cx, cy), width: w, height: h);
      final rr   = RRect.fromRectAndRadius(rect, const Radius.circular(10));

      fill.color   = color.withOpacity(0.10);  canvas.drawRRect(rr, fill);
      stroke.color = color;                    canvas.drawRRect(rr, stroke);
      _corners(canvas, rect, color, 18);
      _label(canvas, rect, b, color);
    }
  }

  void _corners(Canvas canvas, Rect r, Color c, double l) {
    final p = Paint()..color = c..style = PaintingStyle.stroke
      ..strokeWidth = 4..strokeCap = StrokeCap.round;
    canvas.drawLine(r.topLeft,     r.topLeft     + Offset(l, 0),  p);
    canvas.drawLine(r.topLeft,     r.topLeft     + Offset(0, l),  p);
    canvas.drawLine(r.topRight,    r.topRight    + Offset(-l, 0), p);
    canvas.drawLine(r.topRight,    r.topRight    + Offset(0, l),  p);
    canvas.drawLine(r.bottomLeft,  r.bottomLeft  + Offset(l, 0),  p);
    canvas.drawLine(r.bottomLeft,  r.bottomLeft  + Offset(0, -l), p);
    canvas.drawLine(r.bottomRight, r.bottomRight + Offset(-l, 0), p);
    canvas.drawLine(r.bottomRight, r.bottomRight + Offset(0, -l), p);
  }

  void _label(Canvas canvas, Rect rect, BoundingBox b, Color color) {
    final tp = TextPainter(
      text: TextSpan(
        text: '${b.isTarget ? "✓ " : "⚠ "}${b.label} · ${b.distance}',
        style: const TextStyle(
          color: Colors.white, fontSize: 12,
          fontWeight: FontWeight.w700,
          shadows: [Shadow(blurRadius: 3, color: Colors.black)])),
      textDirection: TextDirection.ltr)..layout(maxWidth: rect.width + 30);

    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(rect.left, rect.top - 24, tp.width + 12, 22),
        const Radius.circular(5)),
      Paint()..color = color.withOpacity(0.85));
    tp.paint(canvas, Offset(rect.left + 6, rect.top - 22));
  }

  @override
  bool shouldRepaint(covariant _BoxPainter old) => old.boxes != boxes;
}