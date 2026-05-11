// mobile/lib/text_to_sign_page.dart
// ════════════════════════════════════════════════════════════════
// Text → Sign Language  (avatar MP4 playback)
//
// Flow:
//   1. User types sentence → POST /sign/text_to_sign
//   2. Backend returns segments: word videos first, letter-by-letter
//      fallback if no word video exists (A-Z also have avatar videos).
//   3. Slides play sequentially via VideoPlayerController.networkUrl.
//      Each video's `ended` event triggers the next slide.
// ════════════════════════════════════════════════════════════════

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import 'api_service.dart';

// ── Colours ───────────────────────────────────────────────────────
const _kPrimary   = Color(0xFF2E8BFF);
const _kAccent    = Color(0xFF00C896);
const _kSecondary = Color(0xFF4FC3F7);
const _kBg        = Color(0xFF060E22);
const _kCard      = Color(0xFF0D1B3E);
const _kText      = Colors.white;
const _kSubtext   = Color(0xFF8BA0C8);

// ── Slide model ───────────────────────────────────────────────────
class _Slide {
  final String  display;    // word or letter
  final String? videoUrl;   // null → text-only fallback
  final bool    isLetter;
  final String  parentWord;

  const _Slide({
    required this.display,
    this.videoUrl,
    this.isLetter = false,
    required this.parentWord,
  });
}

// ── Page ──────────────────────────────────────────────────────────
class TextToSignPage extends StatefulWidget {
  const TextToSignPage({super.key});
  @override
  State<TextToSignPage> createState() => _TextToSignPageState();
}

class _TextToSignPageState extends State<TextToSignPage> {

  final _textCtrl  = TextEditingController();
  final _scrollCtrl = ScrollController();

  // API
  bool    _loading = false;
  String? _error;
  List<_Slide> _slides = [];

  // Playback
  int    _idx     = 0;
  bool   _playing = false;
  double _speed   = 1.0;

  Timer?  _fallbackTimer;     // for text-only slides

  VideoPlayerController? _videoCtrl;
  bool _videoReady    = false;
  bool _videoBuffering = false;

  @override
  void dispose() {
    _fallbackTimer?.cancel();
    _videoCtrl?.dispose();
    _textCtrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  // ── API ───────────────────────────────────────────────────────

  Future<void> _convert() async {
    final text = _textCtrl.text.trim();
    if (text.isEmpty) return;
    FocusScope.of(context).unfocus();

    setState(() { _loading = true; _error = null; _slides = []; _idx = 0; });
    _stopPlayback();

    final res = await ApiService.textToSign(text: text);

    if (res.containsKey('error')) {
      setState(() { _loading = false; _error = res['error'].toString(); });
      return;
    }

    final raw = (res['segments'] as List<dynamic>? ?? [])
        .cast<Map<String, dynamic>>();

    final slides = <_Slide>[];
    for (final seg in raw) {
      final token    = seg['token']     as String? ?? '';
      final videoUrl = seg['video_url'] as String?;
      final letters  = (seg['letters'] as List<dynamic>? ?? [])
          .cast<Map<String, dynamic>>();

      if (videoUrl != null) {
        slides.add(_Slide(display: token, videoUrl: videoUrl, parentWord: token));
      } else if (letters.isNotEmpty) {
        for (final l in letters) {
          slides.add(_Slide(
            display:    l['letter'] as String? ?? '',
            videoUrl:   l['video_url'] as String?,
            isLetter:   true,
            parentWord: token,
          ));
        }
      } else {
        slides.add(_Slide(display: token, parentWord: token));
      }
    }

    setState(() { _loading = false; _slides = slides; });
    if (slides.isNotEmpty) _startPlayback();
  }

  // ── Playback ──────────────────────────────────────────────────

  Duration get _textSlideDelay =>
      Duration(milliseconds: (2000 / _speed).round());

  Future<void> _startPlayback() async {
    setState(() { _playing = true; });
    await _loadSlide(_idx);
  }

  void _stopPlayback() {
    _fallbackTimer?.cancel();
    _fallbackTimer = null;
    _videoCtrl?.pause();
    setState(() { _playing = false; _videoBuffering = false; });
  }

  void _togglePlayback() {
    if (_playing) {
      _stopPlayback();
    } else if (_slides.isNotEmpty) {
      setState(() { _playing = true; });
      if (_videoReady && _videoCtrl != null) {
        _videoCtrl!.play();
      } else {
        _loadSlide(_idx);
      }
    }
  }

  Future<void> _loadSlide(int idx) async {
    if (idx < 0 || idx >= _slides.length) {
      setState(() { _playing = false; });
      return;
    }

    _fallbackTimer?.cancel();
    _fallbackTimer = null;

    // Dispose old controller
    final oldCtrl = _videoCtrl;
    _videoCtrl = null;
    setState(() { _idx = idx; _videoReady = false; _videoBuffering = false; });
    await oldCtrl?.dispose();

    final slide = _slides[idx];

    if (slide.videoUrl != null) {
      setState(() { _videoBuffering = true; });
      final ctrl = VideoPlayerController.networkUrl(Uri.parse(slide.videoUrl!));
      _videoCtrl = ctrl;

      try {
        await ctrl.initialize();
        if (!mounted) return;
        await ctrl.setPlaybackSpeed(_speed);
        setState(() { _videoReady = true; _videoBuffering = false; });

        ctrl.addListener(_onVideoTick);
        if (_playing) await ctrl.play();
      } catch (e) {
        print('[sign_video] load error: $e — falling back to text timer');
        if (!mounted) return;
        setState(() { _videoBuffering = false; });
        _scheduleTextAdvance();
      }
    } else {
      _scheduleTextAdvance();
    }
  }

  void _onVideoTick() {
    final ctrl = _videoCtrl;
    if (ctrl == null || !ctrl.value.isInitialized) return;
    if (!ctrl.value.isPlaying &&
        ctrl.value.duration.inMilliseconds > 0 &&
        ctrl.value.position >= ctrl.value.duration - const Duration(milliseconds: 100)) {
      ctrl.removeListener(_onVideoTick);
      if (_playing && mounted) _advance();
    }
  }

  void _scheduleTextAdvance() {
    _fallbackTimer?.cancel();
    _fallbackTimer = Timer(_textSlideDelay, () {
      if (_playing && mounted) _advance();
    });
  }

  void _advance() {
    if (_idx + 1 < _slides.length) {
      _loadSlide(_idx + 1);
    } else {
      setState(() { _playing = false; });
    }
  }

  void _prev() {
    if (_idx > 0) _loadSlide(_idx - 1);
  }

  void _next() {
    if (_idx + 1 < _slides.length) _loadSlide(_idx + 1);
  }

  void _jumpTo(int idx) => _loadSlide(idx);

  // ── Build ─────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final screenH = MediaQuery.of(context).size.height;

    return Scaffold(
      backgroundColor: _kBg,
      body: SafeArea(
        child: Column(children: [
          _buildHeader(context),
          Expanded(
            child: SingleChildScrollView(
              controller: _scrollCtrl,
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildInputCard(),
                  if (_loading) ...[
                    const SizedBox(height: 32),
                    _buildLoading(),
                  ],
                  if (_error != null) ...[
                    const SizedBox(height: 12),
                    _buildError(),
                  ],
                  if (_slides.isNotEmpty) ...[
                    const SizedBox(height: 16),
                    _buildNowSigning(),
                    const SizedBox(height: 10),
                    _buildVideoArea(screenH),
                    const SizedBox(height: 14),
                    _buildControls(),
                    const SizedBox(height: 14),
                    _buildWordList(),
                  ],
                ],
              ),
            ),
          ),
        ]),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────

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
            child: const Icon(Icons.arrow_back_rounded,
                color: Colors.white, size: 20),
          ),
        ),
        const SizedBox(width: 12),
        const Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Text to Sign Language',
                style: TextStyle(color: Colors.white,
                    fontSize: 18, fontWeight: FontWeight.w800)),
            Text('3D avatar · ISL signs + letter-by-letter fallback',
                style: TextStyle(color: Color(0xFFE7ECFF), fontSize: 11)),
          ]),
        ),
        const Text('✋', style: TextStyle(fontSize: 22)),
      ]),
    );
  }

  // ── Input ────────────────────────────────────────────────────

  Widget _buildInputCard() {
    return Container(
      decoration: BoxDecoration(
        color: _kCard,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _kPrimary.withOpacity(0.2)),
      ),
      padding: const EdgeInsets.all(14),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        TextField(
          controller: _textCtrl,
          maxLines: 2,
          style: const TextStyle(color: _kText, fontSize: 15),
          decoration: InputDecoration(
            hintText: 'Type text to sign  (e.g. "Hello my name is Ahmed")',
            hintStyle: TextStyle(color: _kSubtext.withOpacity(0.7), fontSize: 13),
            filled: true,
            fillColor: _kBg,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: BorderSide.none,
            ),
            contentPadding: const EdgeInsets.all(12),
          ),
          textInputAction: TextInputAction.done,
          onSubmitted: (_) => _convert(),
        ),
        const SizedBox(height: 10),
        GestureDetector(
          onTap: _loading ? null : _convert,
          child: Container(
            height: 46,
            decoration: BoxDecoration(
              gradient: const LinearGradient(colors: [_kPrimary, _kAccent]),
              borderRadius: BorderRadius.circular(13),
              boxShadow: [BoxShadow(
                  color: _kPrimary.withOpacity(0.3),
                  blurRadius: 10, offset: const Offset(0, 3))],
            ),
            child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              if (_loading)
                const SizedBox(width: 16, height: 16,
                    child: CircularProgressIndicator(
                        strokeWidth: 2, color: Colors.white))
              else
                const Icon(Icons.sign_language_rounded,
                    color: Colors.white, size: 18),
              const SizedBox(width: 8),
              Text(_loading ? 'Processing…' : 'Convert to Sign Language',
                  style: const TextStyle(color: Colors.white,
                      fontSize: 14, fontWeight: FontWeight.w700)),
            ]),
          ),
        ),
      ]),
    );
  }

  Widget _buildLoading() => const Center(
      child: CircularProgressIndicator(color: _kPrimary));

  Widget _buildError() => Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      color: Colors.red.withOpacity(0.1),
      borderRadius: BorderRadius.circular(12),
      border: Border.all(color: Colors.red.withOpacity(0.3)),
    ),
    child: Row(children: [
      const Icon(Icons.warning_amber_rounded, color: Colors.red, size: 16),
      const SizedBox(width: 8),
      Expanded(child: Text(_error ?? '',
          style: const TextStyle(color: Colors.red, fontSize: 12))),
    ]),
  );

  // ── "Now signing" label ────────────────────────────────────────

  Widget _buildNowSigning() {
    if (_slides.isEmpty) return const SizedBox.shrink();
    final slide  = _slides[_idx];
    final label  = slide.isLetter
        ? 'Spelling: ${slide.parentWord} → ${slide.display}'
        : slide.display;

    return Row(children: [
      Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: _kAccent.withOpacity(0.12),
          borderRadius: BorderRadius.circular(99),
          border: Border.all(color: _kAccent.withOpacity(0.35)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          if (_playing)
            const _BlinkDot()
          else
            const Icon(Icons.sign_language_rounded, color: _kAccent, size: 12),
          const SizedBox(width: 6),
          Text(
            _playing ? 'Signing: $label' : label,
            style: const TextStyle(color: _kAccent,
                fontSize: 13, fontWeight: FontWeight.w800),
          ),
        ]),
      ),
      const SizedBox(width: 8),
      Text('${_idx + 1}/${_slides.length}',
          style: const TextStyle(color: _kSubtext, fontSize: 11)),
    ]);
  }

  // ── Video area ────────────────────────────────────────────────

  Widget _buildVideoArea(double screenH) {
    final h = (screenH * 0.48).clamp(260.0, 420.0);

    return Container(
      height: h,
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: _playing ? _kAccent : _kPrimary.withOpacity(0.35),
          width: _playing ? 2.5 : 1.2,
        ),
      ),
      clipBehavior: Clip.hardEdge,
      child: Stack(fit: StackFit.expand, children: [

        // ── Main content: video or text card ─────────────────
        _buildVideoOrCard(),

        // ── Buffering spinner ────────────────────────────────
        if (_videoBuffering)
          Container(
            color: Colors.black54,
            child: const Center(
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                CircularProgressIndicator(color: _kAccent),
                SizedBox(height: 12),
                Text('Loading avatar…',
                    style: TextStyle(color: Colors.white70, fontSize: 12)),
              ]),
            ),
          ),

        // ── LIVE badge ───────────────────────────────────────
        if (_playing && !_videoBuffering)
          Positioned(
            top: 10, right: 10,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: _kAccent.withOpacity(0.85),
                borderRadius: BorderRadius.circular(99),
              ),
              child: const Row(mainAxisSize: MainAxisSize.min, children: [
                _BlinkDot(size: 6),
                SizedBox(width: 4),
                Text('LIVE', style: TextStyle(
                    color: Colors.white, fontSize: 9,
                    fontWeight: FontWeight.w800)),
              ]),
            ),
          ),
      ]),
    );
  }

  Widget _buildVideoOrCard() {
    final slide    = _slides.isEmpty ? null : _slides[_idx];
    final hasVideo = _videoReady && _videoCtrl != null &&
                     _videoCtrl!.value.isInitialized;

    if (hasVideo) {
      final size = _videoCtrl!.value.size;
      final w    = size.width  > 0 ? size.width  : 480.0;
      final h    = size.height > 0 ? size.height : 360.0;
      return FittedBox(
        fit: BoxFit.contain,
        child: SizedBox(
          width:  w,
          height: h,
          child:  VideoPlayer(_videoCtrl!),
        ),
      );
    }

    // Text fallback — shown while buffering or if no video
    if (_videoBuffering) return const SizedBox.shrink(); // spinner covers this

    if (slide == null) return const SizedBox.shrink();

    return Container(
      decoration: BoxDecoration(
        gradient: RadialGradient(colors: [
          _kPrimary.withOpacity(0.15), Colors.black,
        ]),
      ),
      child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Text(slide.isLetter ? '✋' : '🤟',
            style: const TextStyle(fontSize: 44)),
        const SizedBox(height: 14),
        Text(
          slide.display,
          style: TextStyle(
            color: _kText,
            fontSize: slide.isLetter ? 72 : 48,
            fontWeight: FontWeight.w900,
            letterSpacing: 3,
          ),
        ),
        if (slide.isLetter) ...[
          const SizedBox(height: 6),
          Text('spelling: ${slide.parentWord}',
              style: const TextStyle(color: _kSubtext, fontSize: 12)),
        ],
        const SizedBox(height: 14),
        const Text('Avatar video unavailable',
            style: TextStyle(color: _kSubtext, fontSize: 11)),
      ]),
    );
  }

  // ── Transport controls ────────────────────────────────────────

  Widget _buildControls() {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 10),
      decoration: BoxDecoration(
        color: _kCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _kPrimary.withOpacity(0.12)),
      ),
      child: Column(children: [
        // Progress bar
        if (_slides.isNotEmpty) ...[
          _LinearProgress(current: _idx + 1, total: _slides.length),
          const SizedBox(height: 12),
        ],
        // Transport row
        Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          _CtrlBtn(
            icon: Icons.skip_previous_rounded,
            onTap: _idx > 0 ? _prev : null,
          ),
          const SizedBox(width: 16),
          GestureDetector(
            onTap: _togglePlayback,
            child: Container(
              width: 56, height: 56,
              decoration: BoxDecoration(
                gradient: const LinearGradient(colors: [_kPrimary, _kAccent]),
                shape: BoxShape.circle,
                boxShadow: [BoxShadow(
                    color: _kPrimary.withOpacity(0.4),
                    blurRadius: 12, offset: const Offset(0, 4))],
              ),
              child: Icon(
                _playing ? Icons.pause_rounded : Icons.play_arrow_rounded,
                color: Colors.white, size: 30,
              ),
            ),
          ),
          const SizedBox(width: 16),
          _CtrlBtn(
            icon: Icons.skip_next_rounded,
            onTap: _idx + 1 < _slides.length ? _next : null,
          ),
        ]),
        const SizedBox(height: 10),
        // Speed row
        Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          const Text('Speed ', style: TextStyle(color: _kSubtext, fontSize: 11)),
          for (final s in [0.5, 1.0, 1.5, 2.0])
            Padding(
              padding: const EdgeInsets.only(left: 6),
              child: GestureDetector(
                onTap: () {
                  setState(() => _speed = s);
                  _videoCtrl?.setPlaybackSpeed(s);
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: _speed == s ? _kPrimary : _kPrimary.withOpacity(0.08),
                    borderRadius: BorderRadius.circular(99),
                  ),
                  child: Text('${s}x',
                      style: TextStyle(
                        color: _speed == s ? Colors.white : _kSubtext,
                        fontSize: 11, fontWeight: FontWeight.w700,
                      )),
                ),
              ),
            ),
        ]),
      ]),
    );
  }

  // ── Word chip list (like original app's highlighted word list) ──

  Widget _buildWordList() {
    // Unique parent words in order
    final seen  = <String>{};
    final words = <String>[];
    for (final s in _slides) {
      if (seen.add(s.parentWord)) words.add(s.parentWord);
    }
    final currentWord = _slides.isEmpty ? '' : _slides[_idx].parentWord;

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _kCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _kPrimary.withOpacity(0.12)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Sign Sequence',
            style: TextStyle(color: _kSecondary, fontSize: 12,
                fontWeight: FontWeight.w700, letterSpacing: 0.5)),
        const SizedBox(height: 10),
        Wrap(spacing: 6, runSpacing: 6,
          children: words.map((word) {
            final isCur  = word == currentWord;
            final slideIdx = _slides.indexWhere((s) => s.parentWord == word);
            return GestureDetector(
              onTap: () => _jumpTo(slideIdx),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: isCur ? _kAccent : _kPrimary.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(99),
                  border: Border.all(
                    color: isCur ? _kAccent : _kPrimary.withOpacity(0.2),
                  ),
                ),
                child: Text(word,
                    style: TextStyle(
                      color: isCur ? Colors.black : _kSubtext,
                      fontSize: 13, fontWeight: FontWeight.w700,
                    )),
              ),
            );
          }).toList(),
        ),
      ]),
    );
  }
}

// ── Reusable sub-widgets ──────────────────────────────────────────

class _CtrlBtn extends StatelessWidget {
  final IconData    icon;
  final VoidCallback? onTap;
  const _CtrlBtn({required this.icon, this.onTap});

  @override
  Widget build(BuildContext context) {
    final on = onTap != null;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 44, height: 44,
        decoration: BoxDecoration(
          color: on ? _kPrimary.withOpacity(0.12) : Colors.transparent,
          shape: BoxShape.circle,
        ),
        child: Icon(icon,
            color: on ? _kPrimary : _kSubtext.withOpacity(0.3), size: 26),
      ),
    );
  }
}

class _LinearProgress extends StatelessWidget {
  final int current;
  final int total;
  const _LinearProgress({required this.current, required this.total});

  @override
  Widget build(BuildContext context) {
    final pct = total > 0 ? current / total : 0.0;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        Text('Sign $current of $total',
            style: const TextStyle(color: _kSubtext, fontSize: 11)),
        const Spacer(),
        Text('${(pct * 100).toStringAsFixed(0)}%',
            style: const TextStyle(color: _kSubtext, fontSize: 11)),
      ]),
      const SizedBox(height: 5),
      ClipRRect(
        borderRadius: BorderRadius.circular(99),
        child: LinearProgressIndicator(
          value: pct, minHeight: 4,
          backgroundColor: _kPrimary.withOpacity(0.12),
          valueColor: const AlwaysStoppedAnimation(_kPrimary),
        ),
      ),
    ]);
  }
}

// Animated blinking dot for the LIVE badge
class _BlinkDot extends StatefulWidget {
  final double size;
  const _BlinkDot({this.size = 8});
  @override
  State<_BlinkDot> createState() => _BlinkDotState();
}
class _BlinkDotState extends State<_BlinkDot>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 700),
  )..repeat(reverse: true);

  @override
  void dispose() { _c.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _c,
      child: Container(
        width: widget.size, height: widget.size,
        decoration: const BoxDecoration(
          color: _kAccent, shape: BoxShape.circle,
        ),
      ),
    );
  }
}
