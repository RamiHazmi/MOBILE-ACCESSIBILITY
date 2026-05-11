// ui_home.dart — v7 (Blue Orb UI)
//
// • Animated glowing blue orb (replaces robot widget)
// • Dark navy/deep blue gradient background
// • "Hello Robin" + large greeting text (idle)
// • "Searching..." pill button while processing
// • "Great! Here is what I found" when response received
// • Top bar: deaf-mode button only
// • Bottom card: transcript + response text
// • Live amplitude-driven waveform while recording

import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'recorder.dart';
import 'api_service.dart';
import 'camera_screen.dart';
import 'navigation_screen.dart';
import 'tts_service.dart';
import 'handwriting_screen.dart';
import 'lip_reading_page.dart';
import 'sign_language_page.dart';
import 'gesture_sign_page.dart';
import 'save_object_screen.dart';
import 'save_person_screen.dart';
import 'alarm_service.dart';
import 'alarm_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with TickerProviderStateMixin {
  String _status      = '';
  String _transcript  = '';
  String _speak       = '';
  String _danger      = 'LOW';
  bool   _recording   = false;
  bool   _processing  = false;
  bool   _ready       = false;
  bool   _awaitingConfirmation = false;
  bool   _robotSpeaking        = false;
  bool   _awaitingName         = false;  // true on first launch until name is saved

  // Exclusion zone measurement
  final GlobalKey _topBarKey     = GlobalKey();
  final GlobalKey _bottomCardKey = GlobalKey();
  double _topBarHeight     = 80.0;
  double _bottomCardHeight = 0.0;

  // Orb animations
  late AnimationController _orbRotateAnim;   // slow spin of the streaks
  late AnimationController _orbPulseAnim;    // breathing glow pulse
  late AnimationController _orbShimmerAnim;  // fast shimmer highlight
  late AnimationController _rippleAnim;
  Offset _rippleOrigin = Offset.zero;

  // Amplitude — smoothed value fed to the waveform painter
  double _amplitude = 0.0;
  StreamSubscription<double>? _ampSub;

  static const Set<String> _cameraModes = {'FIND_OBJECT', 'DESCRIBE'};

  @override
  void initState() {
    super.initState();

    _orbRotateAnim = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 6000))
      ..repeat();

    _orbPulseAnim = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 2400))
      ..repeat(reverse: true);

    _orbShimmerAnim = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1500))
      ..repeat();

    _rippleAnim = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 500));

    SystemChrome.setPreferredOrientations(
        [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
    SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarBrightness: Brightness.dark,
      statusBarIconBrightness: Brightness.light,
    ));

    _ampSub = RecorderService.amplitudeStream.listen((raw) {
      if (!mounted) return;
      final alpha = raw > _amplitude ? 0.35 : 0.12;
      setState(() {
        _amplitude = _amplitude + alpha * (raw - _amplitude);
      });
    });

    // ── Start alarm detection service ──────────────────────────
    AlarmService.instance.latestAlert.addListener(_onAlarmUpdate);
    AlarmService.instance.start();

    _boot();
  }

  void _onAlarmUpdate() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _ampSub?.cancel();
    _orbRotateAnim.dispose();
    _orbPulseAnim.dispose();
    _orbShimmerAnim.dispose();
    _rippleAnim.dispose();
    // Keep AlarmService running — don't stop it when home page disposes.
    AlarmService.instance.latestAlert.removeListener(_onAlarmUpdate);
    super.dispose();
  }

  /// Detect the TTS language code best suited to speak a name.
  /// Arabic/Persian/Urdu script → 'ar', Latin → 'en'.
  String _detectNameLang(String name) {
    final arabicChars = name.runes.where((r) =>
        (r >= 0x0600 && r <= 0x06FF) ||  // Arabic
        (r >= 0x0750 && r <= 0x077F) ||  // Arabic Supplement
        (r >= 0xFB50 && r <= 0xFDFF) ||  // Arabic Presentation Forms-A
        (r >= 0xFE70 && r <= 0xFEFF)     // Arabic Presentation Forms-B
    ).length;
    return arabicChars > 0 ? 'ar' : 'en';
  }

  Future<void> _boot() async {
    await TtsService.init();

    // ── Check if we know the user's name ──────────────────────
    final storedName = await ApiService.getUserName();

    if (storedName.isEmpty) {
      // First launch — ask for name
      await _speakWithRobot(
          'Welcome to WExist! What is your name? Tap the screen and tell me.',
          lang: 'en');
      if (mounted) {
        setState(() {
          _ready  = true;
          _status = 'Tap and say your name';
          _awaitingName = true;
        });
      }
    } else {
      // Returning user — greet by name
      // Detect script of the name to pick the right TTS language
      final nameLang = _detectNameLang(storedName);
      await _speakWithRobot('Hey $storedName, how can I help you today?',
          lang: nameLang);
      if (mounted) {
        setState(() {
          _ready  = true;
          _status = 'Tap anywhere to speak';
        });
      }
    }
  }

  void _measureZones() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final topBox =
          _topBarKey.currentContext?.findRenderObject() as RenderBox?;
      if (topBox != null) _topBarHeight = topBox.size.height + 16;

      final botBox =
          _bottomCardKey.currentContext?.findRenderObject() as RenderBox?;
      if (botBox != null) _bottomCardHeight = botBox.size.height + 16;
    });
  }

  void _onScreenTap(TapUpDetails details) {
    if (!_ready || _processing) return;

    final screenH = MediaQuery.of(context).size.height;
    final dy      = details.globalPosition.dy;
    final topPad  = MediaQuery.of(context).padding.top;
    final botPad  = MediaQuery.of(context).padding.bottom;

    if (dy < _topBarHeight + topPad) return;
    if (_bottomCardHeight > 0 &&
        dy > screenH - _bottomCardHeight - botPad) return;

    _rippleOrigin = details.globalPosition;
    _rippleAnim.forward(from: 0);

    HapticFeedback.lightImpact();

    if (!_recording) {
      _startRecording();
    } else {
      _stopAndProcess();
    }
  }

  Future<void> _startRecording() async {
    final path = await RecorderService.startRecording();
    if (path == null) {
      setState(() => _status = 'Microphone permission denied');
      return;
    }
    setState(() {
      _recording  = true;
      _amplitude  = 0.0;
      _status     = _awaitingName
          ? 'Listening for your name…'
          : _awaitingConfirmation
              ? 'Listening for your answer…'
              : 'Listening…';
      _transcript = '';
      _speak      = '';
    });
  }

  Future<void> _stopAndProcess() async {
    setState(() {
      _recording  = false;
      _amplitude  = 0.0;
      _processing = true;
      _status     = 'Thinking…';
    });

    final audioPath = await RecorderService.stopRecording();
    if (audioPath == null) {
      setState(() { _processing = false; _status = 'No audio captured'; });
      return;
    }

    // ── Name capture mode (first launch) ──────────────────────
    if (_awaitingName) {
      // Use the same /process endpoint — we only need the transcription
      final raw  = await ApiService.process(audioPath: audioPath);
      final data = jsonDecode(raw) as Map<String, dynamic>;
      final name = (data['transcription'] as String? ?? '').trim();

      if (name.isNotEmpty) {
        await ApiService.saveUserName(name);
        setState(() {
          _awaitingName = false;
          _processing   = false;
          _status       = 'Tap anywhere to speak';
        });
        final nameLang = _detectNameLang(name);
        await _speakWithRobot('Nice to meet you, $name! How can I help you today?',
            lang: nameLang);
      } else {
        setState(() { _processing = false; _status = 'Tap and say your name'; });
        await _speakWithRobot("I didn't catch that. Please tap and say your name.",
            lang: 'en');
      }
      return;
    }

    final raw = await ApiService.process(audioPath: audioPath);
    _handleResponse(raw);
  }

  Future<void> _speakWithRobot(String text,
      {required String lang, bool interrupt = false}) async {
    if (text.trim().isEmpty) return;
    if (mounted) setState(() => _robotSpeaking = true);
    await TtsService.speak(text, lang: lang, interrupt: interrupt);
    final words  = text.trim().split(RegExp(r'\s+')).length;
    final holdMs = (words * 250).clamp(900, 4200);
    await Future<void>.delayed(Duration(milliseconds: holdMs));
    if (mounted) setState(() => _robotSpeaking = false);
  }

  void _handleResponse(String raw) {
    try {
      final data = jsonDecode(raw) as Map<String, dynamic>;

      if (data.containsKey('error')) {
        setState(() {
          _processing = false;
          _status     = 'Error: ${data['error']}';
        });
        return;
      }

      final mode       = (data['mode']       ?? 'IDLE') as String;
      final speak      = (data['speak']       ?? '')     as String;
      final interrupt  = (data['interrupt']   ?? false)  as bool;
      final danger     = (data['danger']      ?? 'LOW')  as String;
      final transcript = (data['transcription'] ?? '')   as String;
      final needsClar  = (data['needs_clarification'] ?? false) as bool;
      final clarQ      = (data['clarification_question'] ?? '') as String;
      final awaiting   = (data['awaiting_confirmation']  ?? false) as bool;
      final entities   =
          (data['entities'] as List? ?? []).map((e) => e.toString()).toList();
      final language   = (data['language'] ?? 'en') as String;

      setState(() {
        _processing           = false;
        _transcript           = transcript;
        _speak                = speak.isNotEmpty
            ? speak
            : (clarQ.isNotEmpty ? clarQ : '');
        _danger               = danger;
        _awaitingConfirmation = awaiting;
        _status               = awaiting
            ? 'Tap & say YES or NO'
            : (needsClar ? 'Tap and try again' : 'Tap anywhere to speak');
      });

      if (speak.isNotEmpty) {
        _speakWithRobot(speak, lang: language, interrupt: interrupt);
      }

      if (awaiting || needsClar) return;

      if (mode == 'NAVIGATE') {
        final destination = entities.isNotEmpty ? entities.first : '';
        final tl      = transcript.toLowerCase();
        final navMode = (tl.contains('driv') || tl.contains('car') ||
                tl.contains('taxi') || tl.contains('ride'))
            ? 'driving'
            : 'walking';
        final delay = speak.isNotEmpty ? 2200 : 300;
        Future.delayed(Duration(milliseconds: delay), () {
          if (!mounted) return;
          Navigator.of(context)
              .push(MaterialPageRoute(
                builder: (_) => NavigationScreen(
                  destination: destination,
                  navMode: navMode,
                  language: language,
                ),
              ))
              .then((_) => ApiService.reset());
        });
        setState(() => _status =
            destination.isEmpty ? 'Showing your location…' : 'Opening navigation…');
      } else if (mode == 'SAVE_OBJECT') {
        final objectName = entities.isNotEmpty ? entities.first : 'object';
        final delay      = speak.isNotEmpty ? 2200 : 400;
        Future.delayed(Duration(milliseconds: delay), () {
          if (!mounted) return;
          Navigator.of(context)
              .push(MaterialPageRoute(
                builder: (_) => SaveObjectScreen(
                  objectName: objectName,
                  language:   language,
                ),
              ))
              .then((_) => ApiService.reset());
        });
        setState(() => _status = 'Opening camera to save ${objectName}…');
      } else if (mode == 'SAVE_PERSON') {
        final personName   = entities.isNotEmpty ? entities.first : 'person';
        final relationship = entities.length > 1  ? entities[1]  : 'person';
        final delay        = speak.isNotEmpty ? 2200 : 400;
        Future.delayed(Duration(milliseconds: delay), () {
          if (!mounted) return;
          Navigator.of(context)
              .push(MaterialPageRoute(
                builder: (_) => SavePersonScreen(
                  personName:   personName,
                  relationship: relationship,
                  language:     language,
                ),
              ))
              .then((_) => ApiService.reset());
        });
        setState(() => _status = 'Opening camera to save ${personName}…');
      } else if (mode == 'READ_HANDWRITING') {
        final delay = speak.isNotEmpty ? 2200 : 400;
        Future.delayed(Duration(milliseconds: delay), () {
          if (!mounted) return;
          Navigator.of(context)
              .push(MaterialPageRoute(
                builder: (_) => const HandwritingScreen(),
              ))
              .then((_) => ApiService.reset());
        });
        setState(() => _status = 'Opening handwriting reader…');
      } else if (_cameraModes.contains(mode)) {
        final target = entities.isNotEmpty ? entities.first : null;
        final delay  = speak.isNotEmpty ? 2500 : 400;
        Future.delayed(Duration(milliseconds: delay), () {
          if (!mounted) return;
          Navigator.of(context)
              .push(MaterialPageRoute(
                builder: (_) => CameraScreen(
                  mode: mode,
                  findTarget: target,
                  language: language,
                ),
              ))
              .then((_) => ApiService.reset());
        });
        setState(() => _status = 'Opening camera…');
      }
    } catch (e) {
      setState(() { _processing = false; _status = 'Parse error: $e'; });
    }
  }

  // ════════════════════════════════════════════════════════════
  // BUILD
  // ════════════════════════════════════════════════════════════

  @override
  Widget build(BuildContext context) {
    _measureZones();
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Deep blue gradient background ─────────────────────
          _buildBackground(),

          SafeArea(
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTapUp: _onScreenTap,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Column(
                    children: [
                      KeyedSubtree(
                        key: _topBarKey,
                        child: _buildTopModeBar(),
                      ),
                      Expanded(child: _buildCenter()),
                      // Live waveform while recording
                      AnimatedSwitcher(
                        duration: const Duration(milliseconds: 200),
                        child: _recording
                            ? _LiveWaveform(
                                key: const ValueKey('waveform'),
                                amplitude: _amplitude,
                              )
                            : const SizedBox(key: ValueKey('no-waveform'), height: 0),
                      ),
                      KeyedSubtree(
                        key: _bottomCardKey,
                        child: _buildBottomArea(),
                      ),
                      const SizedBox(height: 16),
                    ],
                  ),

                  // ── Tap ripple overlay ─────────────────────────
                  Positioned.fill(
                    child: IgnorePointer(
                      child: AnimatedBuilder(
                        animation: _rippleAnim,
                        builder: (_, __) {
                          if (_rippleAnim.value == 0) {
                            return const SizedBox.shrink();
                          }
                          final size = MediaQuery.of(context).size;
                          final maxR = math.sqrt(
                              size.width * size.width +
                              size.height * size.height);
                          return CustomPaint(
                            painter: _RipplePainter(
                              origin:    _rippleOrigin,
                              progress:  _rippleAnim.value,
                              maxRadius: maxR * 0.6,
                              color:     const Color(0xFF4FC3F7),
                            ),
                          );
                        },
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Background gradient — deep navy like screenshots ────────
  Widget _buildBackground() {
    return AnimatedBuilder(
      animation: _orbPulseAnim,
      builder: (_, __) {
        final pulse = _orbPulseAnim.value; // 0→1→0
        return Container(
          decoration: BoxDecoration(
            gradient: RadialGradient(
              center: const Alignment(0.0, -0.15),
              radius: 1.2,
              colors: [
                Color.lerp(
                  const Color(0xFF0A1A3A),
                  const Color(0xFF0D2560),
                  0.4 + pulse * 0.25,
                )!,
                const Color(0xFF060E22),
                const Color(0xFF020508),
              ],
              stops: const [0.0, 0.55, 1.0],
            ),
          ),
        );
      },
    );
  }

  // ── Top bar — Lip Reading (left) + [alarm badge] + Sign Language (right) ──
  Widget _buildTopModeBar() {
    final alarm = AlarmService.instance.latestAlert.value;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      child: Row(children: [
        _TopNavButton(
          icon: Icons.record_voice_over_rounded,
          label: 'Lip Reading',
          onTap: () => Navigator.of(context).push(
            MaterialPageRoute(builder: (_) => const LipReadingPage()),
          ),
        ),
        const Spacer(),
        // ── Alarm button — always visible, badge when unread ───
        SizedBox(
          width: 82,
          child: alarm != null
            ? _AlarmBadge(
                event: alarm,
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const AlarmPage()),
                ),
              )
            : _AlarmIdleButton(
                hasHistory: AlarmService.instance.history.value.isNotEmpty,
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const AlarmPage()),
                ),
              ),
        ),
        const SizedBox(width: 8),
        _TopNavButton(
          icon: Icons.sign_language_rounded,
          label: 'Sign Language',
          onTap: () => Navigator.of(context).push(
            MaterialPageRoute(builder: (_) => const GestureSignPage()),
          ),
        ),
      ]),
    );
  }

  // ── Center — orb + text ──────────────────────────────────────
  Widget _buildCenter() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // ── Animated Blue Orb ──────────────────────────────
          _AnimatedBlueOrb(
            rotateAnim:  _orbRotateAnim,
            pulseAnim:   _orbPulseAnim,
            shimmerAnim: _orbShimmerAnim,
            isListening: _recording,
            isThinking:  _processing,
            isSpeaking:  _robotSpeaking,
          ),

          const SizedBox(height: 36),

          // ── Text block changes by state ────────────────────
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 400),
            transitionBuilder: (child, anim) => FadeTransition(
              opacity: anim,
              child: SlideTransition(
                position: Tween<Offset>(
                  begin: const Offset(0, 0.08),
                  end: Offset.zero,
                ).animate(anim),
                child: child,
              ),
            ),
            child: _buildCenterText(),
          ),
        ],
      ),
    );
  }

  Widget _buildCenterText() {
    if (_processing) {
      // "Great! Here is what I found" state (processing / just responded)
      return Column(
        key: const ValueKey('processing'),
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'Great!',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white.withOpacity(0.85),
              fontSize: 32,
              fontWeight: FontWeight.w700,
              height: 1.2,
            ),
          ),
          Text(
            'Here is what\nI found',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white.withOpacity(0.55),
              fontSize: 28,
              fontWeight: FontWeight.w400,
              height: 1.3,
            ),
          ),
        ],
      );
    }

    if (_speak.isNotEmpty && !_recording) {
      // After response received
      return Column(
        key: const ValueKey('responded'),
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'Great!',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white.withOpacity(0.85),
              fontSize: 32,
              fontWeight: FontWeight.w700,
              height: 1.2,
            ),
          ),
          Text(
            'Here is what\nI found',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white.withOpacity(0.55),
              fontSize: 28,
              fontWeight: FontWeight.w400,
              height: 1.3,
            ),
          ),
        ],
      );
    }

    if (_recording) {
      return Column(
        key: const ValueKey('listening'),
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'Listening…',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white.withOpacity(0.7),
              fontSize: 22,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      );
    }

    // Idle state — orb only, no hardcoded text (TTS already speaks the greeting)
    return const SizedBox.shrink(key: ValueKey('idle'));
  }

  // ── Bottom area — searching pill or transcript card ──────────
  Widget _buildBottomArea() {
    if (_processing) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(40, 0, 40, 8),
        child: _SearchingPill(),
      );
    }
    return _buildBottomCard();
  }

  Widget _buildBottomCard() {
    if (_transcript.isEmpty && _speak.isEmpty) {
      // Idle: show the mic input hint pill
      return Padding(
        padding: const EdgeInsets.fromLTRB(24, 0, 24, 8),
        child: _InputHintPill(),
      );
    }
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 12),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.06),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.10), width: 1),
        boxShadow: [
          BoxShadow(
              color: Colors.black.withOpacity(0.5),
              blurRadius: 24,
              offset: const Offset(0, 8))
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_transcript.isNotEmpty) ...[
            Row(children: [
              Icon(Icons.record_voice_over_rounded,
                  color: Colors.white.withOpacity(0.25), size: 13),
              const SizedBox(width: 6),
              Expanded(
                child: Text('"$_transcript"',
                    style: TextStyle(
                        color: Colors.white.withOpacity(0.3),
                        fontSize: 12,
                        fontStyle: FontStyle.italic),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis),
              ),
            ]),
            if (_speak.isNotEmpty)
              Divider(color: Colors.white.withOpacity(0.08), height: 18),
          ],
          if (_speak.isNotEmpty)
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Icon(Icons.volume_up_rounded,
                  color: const Color(0xFF4FC3F7).withOpacity(0.75), size: 15),
              const SizedBox(width: 8),
              Expanded(
                child: Text(_speak,
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 15,
                        height: 1.55,
                        fontWeight: FontWeight.w400)),
              ),
            ]),
        ],
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════
// Alarm idle button — always visible when no unread alert
// ════════════════════════════════════════════════════════════

class _AlarmIdleButton extends StatelessWidget {
  final bool         hasHistory;
  final VoidCallback onTap;
  const _AlarmIdleButton({required this.hasHistory, required this.onTap});

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: onTap,
    child: Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 5),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.08),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white.withOpacity(0.2), width: 1.5),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(Icons.notifications_outlined,
            color: Colors.white.withOpacity(0.6), size: 14),
        const SizedBox(width: 4),
        Flexible(child: Column(crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min, children: [
          Text('SOUNDS', style: TextStyle(
              color: Colors.white.withOpacity(0.5),
              fontSize: 7, fontWeight: FontWeight.w800, letterSpacing: 0.8)),
          Text(hasHistory ? 'View history' : 'Monitoring…',
              style: TextStyle(color: Colors.white.withOpacity(0.7),
                  fontSize: 9, fontWeight: FontWeight.w600),
              maxLines: 1, overflow: TextOverflow.ellipsis),
        ])),
      ]),
    ),
  );
}

// ════════════════════════════════════════════════════════════
// Alarm danger badge — shown in the top bar when YAMNet fires
// ════════════════════════════════════════════════════════════

class _AlarmBadge extends StatefulWidget {
  final AlarmEvent   event;
  final VoidCallback onTap;
  const _AlarmBadge({required this.event, required this.onTap});
  @override
  State<_AlarmBadge> createState() => _AlarmBadgeState();
}

class _AlarmBadgeState extends State<_AlarmBadge>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulse;
  late Animation<double>   _anim;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 700))
      ..repeat(reverse: true);
    _anim = Tween<double>(begin: 0.6, end: 1.0).animate(
        CurvedAnimation(parent: _pulse, curve: Curves.easeInOut));
  }

  @override
  void dispose() { _pulse.dispose(); super.dispose(); }

  Color get _catColor {
    switch (widget.event.category) {
      case 'emergency': return const Color(0xFFFF3B30);
      case 'fire':      return const Color(0xFFFF6B35);
      case 'door':      return const Color(0xFF4C6FFF);
      case 'call':      return const Color(0xFF00AA55);
      case 'danger':    return const Color(0xFFFF3B30);
      case 'care':      return const Color(0xFFFF8A65);
      case 'alarm':     return const Color(0xFFFFCC00);
      default:          return const Color(0xFFFF3B30);
    }
  }

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: widget.onTap,
    child: AnimatedBuilder(
      animation: _anim,
      builder: (_, __) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 5),
        decoration: BoxDecoration(
          color:        _catColor.withOpacity(0.15 + _anim.value * 0.05),
          borderRadius: BorderRadius.circular(10),
          border:       Border.all(color: _catColor.withOpacity(0.5 + _anim.value * 0.3), width: 1.5),
          boxShadow: [
            BoxShadow(color: _catColor.withOpacity(0.3 * _anim.value),
                blurRadius: 8, spreadRadius: 1),
          ],
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Text(widget.event.emoji, style: const TextStyle(fontSize: 12)),
          const SizedBox(width: 4),
          Flexible(child: Column(crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min, children: [
            Text('ALERT', style: TextStyle(color: _catColor,
                fontSize: 7, fontWeight: FontWeight.w800, letterSpacing: 0.8)),
            Text(widget.event.description,
                style: const TextStyle(color: Colors.white,
                    fontSize: 10, fontWeight: FontWeight.w600),
                maxLines: 1, overflow: TextOverflow.ellipsis),
          ])),
        ]),
      ),
    ),
  );
}

// ════════════════════════════════════════════════════════════
// Animated Blue Orb — the central hero of the UI
// ════════════════════════════════════════════════════════════

class _AnimatedBlueOrb extends StatelessWidget {
  final AnimationController rotateAnim;
  final AnimationController pulseAnim;
  final AnimationController shimmerAnim;
  final bool isListening;
  final bool isThinking;
  final bool isSpeaking;

  const _AnimatedBlueOrb({
    required this.rotateAnim,
    required this.pulseAnim,
    required this.shimmerAnim,
    required this.isListening,
    required this.isThinking,
    required this.isSpeaking,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: Listenable.merge([rotateAnim, pulseAnim, shimmerAnim]),
      builder: (_, __) {
        final pulse    = pulseAnim.value;      // 0→1
        final rotate   = rotateAnim.value;     // 0→1
        final shimmer  = shimmerAnim.value;    // 0→1

        // Scale breathes slightly
        final scale = isListening
            ? 1.08 + pulse * 0.06
            : 1.0 + pulse * 0.04;

        // Glow intensity increases when speaking/listening
        final glowAlpha = isListening || isSpeaking
            ? 0.55 + pulse * 0.25
            : 0.30 + pulse * 0.15;

        return Transform.scale(
          scale: scale,
          child: SizedBox(
            width: 200,
            height: 200,
            child: CustomPaint(
              painter: _OrbPainter(
                rotate:      rotate,
                shimmer:     shimmer,
                pulse:       pulse,
                glowAlpha:   glowAlpha,
                isListening: isListening,
                isThinking:  isThinking,
              ),
            ),
          ),
        );
      },
    );
  }
}

class _OrbPainter extends CustomPainter {
  final double rotate;
  final double shimmer;
  final double pulse;
  final double glowAlpha;
  final bool   isListening;
  final bool   isThinking;

  const _OrbPainter({
    required this.rotate,
    required this.shimmer,
    required this.pulse,
    required this.glowAlpha,
    required this.isListening,
    required this.isThinking,
  });

  static const Color _coreBlue    = Color(0xFF0A2A6E);
  static const Color _midBlue     = Color(0xFF1550C8);
  static const Color _brightBlue  = Color(0xFF4FC3F7);
  static const Color _white       = Color(0xFFFFFFFF);
  static const Color _accentCyan  = Color(0xFF00E5FF);

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;
    final r  = size.width * 0.44;

    // ── 1. Outer glow halo ────────────────────────────────────
    for (int i = 4; i >= 1; i--) {
      final haloR = r * (1.0 + i * 0.18);
      canvas.drawCircle(
        Offset(cx, cy),
        haloR,
        Paint()
          ..color    = _brightBlue.withOpacity(glowAlpha * 0.08 * (5 - i))
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 28),
      );
    }

    // ── 2. Base sphere gradient ───────────────────────────────
    final spherePaint = Paint()
      ..shader = RadialGradient(
        center: const Alignment(-0.3, -0.3),
        radius: 1.0,
        colors: [
          Color.lerp(_midBlue, _brightBlue, 0.3 + pulse * 0.15)!,
          _coreBlue,
          const Color(0xFF030D24),
        ],
        stops: const [0.0, 0.55, 1.0],
      ).createShader(Rect.fromCircle(center: Offset(cx, cy), radius: r));

    canvas.drawCircle(Offset(cx, cy), r, spherePaint);

    // ── 3. Rotating light streaks (the swirl effect) ─────────
    canvas.save();
    canvas.clipPath(Path()..addOval(Rect.fromCircle(center: Offset(cx, cy), radius: r)));

    _drawStreak(canvas, cx, cy, r,
      angle:   rotate * math.pi * 2,
      arcSpan: math.pi * 1.1,
      width:   5.0,
      color:   _white,
      opacity: 0.90,
      blur:    4.0,
    );

    _drawStreak(canvas, cx, cy, r,
      angle:   rotate * math.pi * 2 + math.pi * 0.6,
      arcSpan: math.pi * 0.75,
      width:   3.0,
      color:   _accentCyan,
      opacity: 0.70,
      blur:    6.0,
    );

    _drawStreak(canvas, cx, cy, r,
      angle:   rotate * math.pi * 2 * 1.4 + math.pi,
      arcSpan: math.pi * 0.55,
      width:   2.0,
      color:   _brightBlue,
      opacity: 0.55,
      blur:    3.0,
    );

    // ── 4. Inner volume depth bands ───────────────────────────
    _drawBand(canvas, cx, cy, r,
      phase:   rotate * math.pi * 2 * 0.7,
      color:   _brightBlue.withOpacity(0.18),
    );

    // ── 5. Shimmer highlight spot ─────────────────────────────
    final shimX = cx - r * 0.28 + math.sin(shimmer * math.pi * 2) * r * 0.10;
    final shimY = cy - r * 0.32 + math.cos(shimmer * math.pi * 2) * r * 0.05;
    canvas.drawCircle(
      Offset(shimX, shimY),
      r * 0.22,
      Paint()
        ..color = _white.withOpacity(0.22 + pulse * 0.10)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 14),
    );

    canvas.restore();

    // ── 6. Rim light (edge glow) ─────────────────────────────
    canvas.drawCircle(
      Offset(cx, cy),
      r,
      Paint()
        ..color = _brightBlue.withOpacity(0.25 + pulse * 0.12)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6),
    );
  }

  void _drawStreak(
    Canvas canvas,
    double cx, double cy, double r, {
    required double angle,
    required double arcSpan,
    required double width,
    required Color  color,
    required double opacity,
    required double blur,
  }) {
    // Draw a curved arc streak that follows the sphere's surface
    final rect = Rect.fromCircle(center: Offset(cx, cy), radius: r * 0.88);
    final paint = Paint()
      ..color = color.withOpacity(opacity)
      ..style = PaintingStyle.stroke
      ..strokeWidth = width
      ..strokeCap = StrokeCap.round
      ..maskFilter = MaskFilter.blur(BlurStyle.normal, blur);

    canvas.drawArc(rect, angle, arcSpan, false, paint);

    // Brighter inner core of the streak
    canvas.drawArc(
      rect,
      angle + arcSpan * 0.3,
      arcSpan * 0.35,
      false,
      Paint()
        ..color = _white.withOpacity(opacity * 0.7)
        ..style = PaintingStyle.stroke
        ..strokeWidth = width * 0.4
        ..strokeCap = StrokeCap.round,
    );
  }

  void _drawBand(
    Canvas canvas,
    double cx, double cy, double r, {
    required double phase,
    required Color  color,
  }) {
    final path = Path();
    const steps = 120;
    for (int i = 0; i <= steps; i++) {
      final t  = i / steps;
      final a  = t * math.pi * 2 + phase;
      final px = cx + (r * 0.82) * math.cos(a);
      final py = cy + (r * 0.30) * math.sin(a);
      if (i == 0) path.moveTo(px, py); else path.lineTo(px, py);
    }
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.2
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3),
    );
  }

  @override
  bool shouldRepaint(_OrbPainter old) =>
      old.rotate  != rotate  ||
      old.shimmer != shimmer ||
      old.pulse   != pulse;
}

// ════════════════════════════════════════════════════════════
// Searching... pill (shown while processing)
// ════════════════════════════════════════════════════════════

class _SearchingPill extends StatefulWidget {
  @override
  State<_SearchingPill> createState() => _SearchingPillState();
}

class _SearchingPillState extends State<_SearchingPill>
    with SingleTickerProviderStateMixin {
  late AnimationController _dotAnim;

  @override
  void initState() {
    super.initState();
    _dotAnim = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 900))
      ..repeat();
  }

  @override
  void dispose() {
    _dotAnim.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _dotAnim,
      builder: (_, __) {
        final dotIndex = (_dotAnim.value * 3).floor();
        final dots = List.generate(3, (i) {
          final opacity = i == dotIndex ? 1.0 : 0.35;
          return Text('•',
              style: TextStyle(
                  color: Colors.white.withOpacity(opacity), fontSize: 18));
        });

        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 16),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.08),
            borderRadius: BorderRadius.circular(40),
            border: Border.all(color: Colors.white.withOpacity(0.12), width: 1),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ...dots.map((d) => Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 2),
                    child: d,
                  )),
              const SizedBox(width: 12),
              const Text(
                'Searching...',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 17,
                  fontWeight: FontWeight.w500,
                  letterSpacing: 0.3,
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

// ════════════════════════════════════════════════════════════
// Input hint pill (idle bottom area — mic cursor)
// ════════════════════════════════════════════════════════════

class _InputHintPill extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.07),
        borderRadius: BorderRadius.circular(40),
        border: Border.all(color: Colors.white.withOpacity(0.10), width: 1),
      ),
      child: Row(
        children: [
          Container(
            width: 2,
            height: 18,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.7),
              borderRadius: BorderRadius.circular(1),
            ),
          ),
          const Spacer(),
          Icon(
            Icons.graphic_eq_rounded,
            color: Colors.white.withOpacity(0.45),
            size: 22,
          ),
        ],
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════
// Live amplitude-driven waveform
// ════════════════════════════════════════════════════════════

class _LiveWaveform extends StatefulWidget {
  final double amplitude;
  const _LiveWaveform({super.key, required this.amplitude});

  @override
  State<_LiveWaveform> createState() => _LiveWaveformState();
}

class _LiveWaveformState extends State<_LiveWaveform>
    with SingleTickerProviderStateMixin {

  late AnimationController _ticker;

  @override
  void initState() {
    super.initState();
    _ticker = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    )..repeat();
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(0, 4, 0, 8),
      child: SizedBox(
        height: 56,
        width: double.infinity,
        child: AnimatedBuilder(
          animation: _ticker,
          builder: (_, __) => CustomPaint(
            painter: _SineWavePainter(
              t:         _ticker.value,
              amplitude: widget.amplitude,
            ),
          ),
        ),
      ),
    );
  }
}

class _SineWavePainter extends CustomPainter {
  final double t;
  final double amplitude;

  static const Color _waveBlue      = Color(0xFF2979FF);
  static const Color _waveBlueLight = Color(0xFF82B1FF);

  const _SineWavePainter({required this.t, required this.amplitude});

  @override
  void paint(Canvas canvas, Size size) {
    final midY  = size.height / 2;
    final amp   = size.height * 0.08 + amplitude * size.height * 0.38;
    final speed = 2.0 + amplitude * 2.5;
    final phase = t * math.pi * 2 * speed;

    _drawWave(
      canvas: canvas, size: size, midY: midY, amp: amp,
      phase: phase, freq: 1.6, color: _waveBlue,
      strokeW: 2.2, opacity: 1.0,
    );
    _drawWave(
      canvas: canvas, size: size, midY: midY, amp: amp * 0.55,
      phase: phase + math.pi * 0.6, freq: 2.2, color: _waveBlueLight,
      strokeW: 1.4, opacity: 0.55,
    );
    _drawWaveFill(
      canvas: canvas, size: size, midY: midY, amp: amp,
      phase: phase, freq: 1.6, color: _waveBlue,
    );
  }

  void _drawWave({
    required Canvas canvas, required Size size,
    required double midY, required double amp,
    required double phase, required double freq,
    required Color color, required double strokeW, required double opacity,
  }) {
    final paint = Paint()
      ..color       = color.withOpacity(opacity)
      ..strokeWidth = strokeW
      ..style       = PaintingStyle.stroke
      ..strokeCap   = StrokeCap.round
      ..strokeJoin  = StrokeJoin.round;

    final path = Path();
    const steps = 200;
    for (int i = 0; i <= steps; i++) {
      final x = size.width * i / steps;
      final y = midY + amp * math.sin(freq * math.pi * 2 * (i / steps) + phase);
      if (i == 0) path.moveTo(x, y); else path.lineTo(x, y);
    }
    canvas.drawPath(path, paint);
  }

  void _drawWaveFill({
    required Canvas canvas, required Size size,
    required double midY, required double amp,
    required double phase, required double freq, required Color color,
  }) {
    final path = Path();
    const steps = 200;
    path.moveTo(0, midY);
    for (int i = 0; i <= steps; i++) {
      final x = size.width * i / steps;
      final y = midY + amp * math.sin(freq * math.pi * 2 * (i / steps) + phase);
      path.lineTo(x, y);
    }
    path.lineTo(size.width, midY);
    path.close();
    canvas.drawPath(
      path,
      Paint()
        ..shader = LinearGradient(
          begin: Alignment.topCenter, end: Alignment.bottomCenter,
          colors: [color.withOpacity(0.18), color.withOpacity(0.0)],
        ).createShader(Rect.fromLTWH(0, 0, size.width, size.height))
        ..style = PaintingStyle.fill,
    );
  }

  @override
  bool shouldRepaint(_SineWavePainter old) =>
      old.t != t || old.amplitude != amplitude;
}

// ════════════════════════════════════════════════════════════
// Top nav button (Lip Reading / Sign Language)
// ════════════════════════════════════════════════════════════

class _TopNavButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  const _TopNavButton({required this.icon, required this.label, required this.onTap});

  static const Color _blue = Color(0xFF4FC3F7);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 42,
        padding: const EdgeInsets.symmetric(horizontal: 14),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(999),
          color: Colors.white.withOpacity(0.06),
          border: Border.all(color: _blue.withOpacity(0.45), width: 1),
          boxShadow: [
            BoxShadow(
              color: _blue.withOpacity(0.15),
              blurRadius: 10,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: _blue, size: 18),
            const SizedBox(width: 8),
            Text(
              label,
              style: const TextStyle(
                color: _blue,
                fontSize: 12,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.3,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ════════════════════════════════════════════════════════════
// Tap ripple
// ════════════════════════════════════════════════════════════

class _RipplePainter extends CustomPainter {
  final Offset origin;
  final double progress;
  final double maxRadius;
  final Color  color;
  const _RipplePainter({
    required this.origin, required this.progress,
    required this.maxRadius, required this.color,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final radius  = maxRadius * progress;
    final opacity = (1.0 - progress) * 0.15;
    canvas.drawCircle(
      origin, radius,
      Paint()
        ..color = color.withOpacity(opacity)
        ..style = PaintingStyle.fill,
    );
  }

  @override
  bool shouldRepaint(_RipplePainter old) => old.progress != progress;
}