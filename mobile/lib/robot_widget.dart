import 'dart:math' as math;
import 'package:flutter/material.dart';

/// Animated robot companion shown on the home screen.
/// Pseudo-3D movement: orbit drift, perspective turns, and dynamic facing.
class RobotWidget extends StatefulWidget {
  final bool isSpeaking;
  final bool isListening;
  final bool isThinking;
  final Color accentColor;
  final double size;

  const RobotWidget({
    super.key,
    this.isSpeaking = false,
    this.isListening = false,
    this.isThinking = false,
    this.accentColor = const Color(0xFF00E5C0),
    this.size = 160,
  });

  @override
  State<RobotWidget> createState() => _RobotWidgetState();
}

class _RobotWidgetState extends State<RobotWidget>
    with TickerProviderStateMixin {
  late AnimationController _blinkCtrl;
  late AnimationController _floatCtrl;
  late AnimationController _mouthCtrl;
  late AnimationController _waveCtrl;
  late AnimationController _thinkCtrl;
  late AnimationController _orbitCtrl;
  late AnimationController _lookCtrl;

  late Animation<double> _blinkAnim;
  late Animation<double> _floatAnim;
  late Animation<double> _mouthAnim;
  late Animation<double> _waveAnim;
  late Animation<double> _thinkAnim;
  late Animation<double> _orbitAnim;
  late Animation<double> _lookAnim;

  @override
  void initState() {
    super.initState();

    _floatCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2400),
    )..repeat(reverse: true);
    _floatAnim = Tween<double>(begin: -6, end: 6).animate(
      CurvedAnimation(parent: _floatCtrl, curve: Curves.easeInOut),
    );

    _blinkCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 150),
    );
    _blinkAnim = Tween<double>(begin: 1, end: 0.05).animate(
      CurvedAnimation(parent: _blinkCtrl, curve: Curves.easeInOut),
    );
    _scheduleBlink();

    _mouthCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 180),
    );
    _mouthAnim = Tween<double>(begin: 0.2, end: 1.0).animate(
      CurvedAnimation(parent: _mouthCtrl, curve: Curves.easeInOut),
    );

    _waveCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
    _waveAnim = Tween<double>(begin: 0, end: 0.4).animate(
      CurvedAnimation(parent: _waveCtrl, curve: Curves.easeInOut),
    );

    _thinkCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    );
    _thinkAnim = Tween<double>(begin: 0, end: 1).animate(_thinkCtrl);

    _orbitCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 7200),
    )..repeat();
    _orbitAnim = Tween<double>(begin: 0, end: 1).animate(_orbitCtrl);

    _lookCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 550),
    );
    _lookAnim = Tween<double>(begin: 0, end: 0).animate(
      CurvedAnimation(parent: _lookCtrl, curve: Curves.easeInOut),
    );
  }

  @override
  void didUpdateWidget(RobotWidget old) {
    super.didUpdateWidget(old);

    if (widget.isSpeaking && !old.isSpeaking) {
      _animateLookTo(0.0);
      _mouthCtrl.repeat(reverse: true);
      _waveCtrl.forward().then((_) => _waveCtrl.reverse());
    } else if (!widget.isSpeaking && old.isSpeaking) {
      _mouthCtrl.stop();
      _mouthCtrl.value = 0;
      _animateLookTo(0.18);
    }

    if (widget.isThinking && !old.isThinking) {
      _thinkCtrl.repeat();
      _animateLookTo(-0.2);
    } else if (!widget.isThinking && old.isThinking) {
      _thinkCtrl.stop();
      _animateLookTo(0.0);
    }

    if (widget.isListening && !old.isListening) {
      _animateLookTo(0.0);
    } else if (!widget.isListening && old.isListening && !widget.isSpeaking) {
      _animateLookTo(0.18);
    }
  }

  void _animateLookTo(double target) {
    _lookAnim = Tween<double>(begin: _lookAnim.value, end: target).animate(
      CurvedAnimation(parent: _lookCtrl, curve: Curves.easeInOut),
    );
    _lookCtrl
      ..stop()
      ..reset()
      ..forward();
  }

  void _scheduleBlink() async {
    while (mounted) {
      await Future<void>.delayed(
          Duration(milliseconds: 2500 + math.Random().nextInt(2000)));
      if (!mounted) break;
      await _blinkCtrl.forward();
      await Future<void>.delayed(const Duration(milliseconds: 120));
      if (!mounted) break;
      await _blinkCtrl.reverse();
    }
  }

  @override
  void dispose() {
    _blinkCtrl.dispose();
    _floatCtrl.dispose();
    _mouthCtrl.dispose();
    _waveCtrl.dispose();
    _thinkCtrl.dispose();
    _orbitCtrl.dispose();
    _lookCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: Listenable.merge([
        _floatAnim,
        _blinkAnim,
        _mouthAnim,
        _waveAnim,
        _thinkAnim,
        _orbitAnim,
        _lookAnim
      ]),
      builder: (_, __) {
        final orbital = math.pi * 2 * _orbitAnim.value;
        final idleYaw = math.sin(orbital) * 0.32;
        final isEngaged = widget.isSpeaking || widget.isListening;
        final yaw = (isEngaged ? 0.0 : idleYaw) + _lookAnim.value;
        final pitch = math.sin(orbital * 0.7) * 0.07;
        final driftX = math.sin(orbital) * 14;
        final depthScale = 0.92 + math.cos(orbital) * 0.06;

        return Transform.translate(
          offset: Offset(driftX, _floatAnim.value),
          child: SizedBox(
            width: widget.size,
            height: widget.size * 1.3,
            child: Stack(
              alignment: Alignment.center,
              children: [
                Positioned(
                  bottom: widget.size * 0.02,
                  child: Transform.scale(
                    scaleX: 1.0 + depthScale * 0.22,
                    scaleY: 0.9,
                    child: Container(
                      width: widget.size * 0.58,
                      height: widget.size * 0.1,
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.22),
                        borderRadius: BorderRadius.circular(999),
                        boxShadow: const [
                          BoxShadow(
                            color: Color(0x55000000),
                            blurRadius: 16,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                Transform(
                  alignment: Alignment.center,
                  transform: Matrix4.identity()
                    ..setEntry(3, 2, 0.0012)
                    ..rotateX(pitch)
                    ..rotateY(yaw)
                    ..scale(depthScale),
                  child: SizedBox(
                    width: widget.size,
                    height: widget.size * 1.3,
                    child: CustomPaint(
                      painter: _RobotPainter(
                        blinkScale: _blinkAnim.value,
                        mouthOpen: _mouthAnim.value,
                        waveAngle: _waveAnim.value,
                        thinkAngle: _thinkAnim.value,
                        accent: widget.accentColor,
                        isSpeaking: widget.isSpeaking,
                        isListening: widget.isListening,
                        isThinking: widget.isThinking,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _RobotPainter extends CustomPainter {
  final double blinkScale;
  final double mouthOpen;
  final double waveAngle;
  final double thinkAngle;
  final Color accent;
  final bool isSpeaking;
  final bool isListening;
  final bool isThinking;

  const _RobotPainter({
    required this.blinkScale,
    required this.mouthOpen,
    required this.waveAngle,
    required this.thinkAngle,
    required this.accent,
    required this.isSpeaking,
    required this.isListening,
    required this.isThinking,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;
    final r = size.width * 0.38;

    final bodyPaint = Paint()..color = const Color(0xFFD8E8F0);
    final darkPaint = Paint()..color = const Color(0xFF1A2A3A);
    final accentPaint = Paint()..color = accent;
    final glowPaint = Paint()
      ..color = accent.withOpacity(0.25)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 12);

    canvas.drawCircle(Offset(cx, cy + r * 0.1), r * 1.15, glowPaint);

    final bodyRect = Rect.fromCenter(
      center: Offset(cx, cy + r * 0.4),
      width: r * 1.5,
      height: r * 1.7,
    );
    canvas.drawOval(bodyRect, bodyPaint);
    canvas.drawOval(
      Rect.fromCenter(
        center: Offset(cx - r * 0.22, cy + r * 0.15),
        width: r * 0.44,
        height: r * 0.9,
      ),
      Paint()..color = Colors.white.withOpacity(0.16),
    );

    final chestRect = Rect.fromCenter(
      center: Offset(cx, cy + r * 0.5),
      width: r * 0.8,
      height: r * 0.6,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(chestRect, const Radius.circular(6)),
      Paint()..color = const Color(0xFFB8CCD8),
    );
    canvas.drawCircle(
      Offset(cx, cy + r * 0.42),
      r * 0.10,
      Paint()..color = accent.withOpacity(0.9),
    );
    canvas.drawCircle(
      Offset(cx, cy + r * 0.42),
      r * 0.10,
      Paint()
        ..color = accent.withOpacity(0.4)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6),
    );

    final headRect = Rect.fromCenter(
      center: Offset(cx, cy - r * 0.2),
      width: r * 1.6,
      height: r * 1.2,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(headRect, Radius.circular(r * 0.25)),
      bodyPaint,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(
          center: Offset(cx - r * 0.2, cy - r * 0.26),
          width: r * 0.5,
          height: r * 0.7,
        ),
        Radius.circular(r * 0.18),
      ),
      Paint()..color = Colors.white.withOpacity(0.14),
    );

    final antennaBase = Rect.fromCenter(
      center: Offset(cx, cy - r * 0.77),
      width: r * 0.22,
      height: r * 0.16,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(antennaBase, const Radius.circular(4)),
      Paint()..color = const Color(0xFFB8CCD8),
    );
    canvas.drawCircle(Offset(cx, cy - r * 0.90), r * 0.09, accentPaint);
    canvas.drawCircle(
      Offset(cx, cy - r * 0.90),
      r * 0.09,
      Paint()
        ..color = accent.withOpacity(0.5)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 5),
    );

    final eyeY = cy - r * 0.22;
    final eyeSpacing = r * 0.38;
    final eyeW = r * 0.28;
    final eyeH = r * 0.22 * blinkScale;

    for (final sign in [-1.0, 1.0]) {
      final ex = cx + sign * eyeSpacing;
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromCenter(
              center: Offset(ex, eyeY), width: eyeW, height: eyeH + 4),
          Radius.circular(eyeW / 2),
        ),
        darkPaint,
      );
      if (blinkScale > 0.3) {
        canvas.drawRRect(
          RRect.fromRectAndRadius(
            Rect.fromCenter(
              center: Offset(ex, eyeY),
              width: eyeW * 0.72,
              height: eyeH * 0.72,
            ),
            Radius.circular(eyeW / 3),
          ),
          Paint()..color = accent,
        );
        canvas.drawCircle(
          Offset(ex + eyeW * 0.12, eyeY - eyeH * 0.15),
          r * 0.04,
          Paint()..color = Colors.white.withOpacity(0.85),
        );
      }
    }

    final mouthY = cy + r * 0.12;
    final mouthW = r * 0.55;
    final mouthH = r * 0.16 * mouthOpen + r * 0.06;
    final mouthRect = Rect.fromCenter(
      center: Offset(cx, mouthY),
      width: mouthW,
      height: mouthH,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(mouthRect, Radius.circular(mouthH / 2)),
      darkPaint,
    );
    if (isSpeaking) {
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromCenter(
            center: Offset(cx, mouthY),
            width: mouthW * 0.7,
            height: mouthH * 0.5,
          ),
          Radius.circular(mouthH / 4),
        ),
        Paint()..color = accent.withOpacity(0.6),
      );
    }

    for (final sign in [-1.0, 1.0]) {
      canvas.drawCircle(
        Offset(cx + sign * r * 0.52, cy + r * 0.08),
        r * 0.10,
        Paint()..color = const Color(0xFFFFB5C8).withOpacity(0.55),
      );
    }

    _drawArm(canvas, cx, cy, r, -1, waveAngle);
    _drawArm(canvas, cx, cy, r, 1, waveAngle);

    final legPaint = Paint()..color = const Color(0xFFB8CCD8);
    final legW = r * 0.28;
    final legH = r * 0.36;
    for (final sign in [-1.0, 1.0]) {
      final lx = cx + sign * r * 0.32;
      final ly = cy + r * 1.26;
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromCenter(center: Offset(lx, ly), width: legW, height: legH),
          Radius.circular(legW / 2),
        ),
        legPaint,
      );
      canvas.drawOval(
        Rect.fromCenter(
          center: Offset(lx + sign * r * 0.06, ly + legH / 2 - r * 0.04),
          width: legW * 1.3,
          height: r * 0.18,
        ),
        legPaint,
      );
    }

    if (isThinking) {
      for (int i = 0; i < 3; i++) {
        final angle = thinkAngle * math.pi * 2 + i * math.pi * 2 / 3;
        final tx = cx + r * 0.9 * math.cos(angle);
        final ty = (cy - r * 0.85) + r * 0.25 * math.sin(angle);
        canvas.drawCircle(
          Offset(tx, ty),
          r * 0.065,
          Paint()..color = accent.withOpacity(0.7 + 0.3 * math.sin(angle)),
        );
      }
    }

    if (isListening) {
      final pulse = Paint()
        ..color = const Color(0xFFFF3B6B).withOpacity(0.7)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);
      canvas.drawCircle(Offset(cx, cy - r * 0.85), r * 0.14, pulse);
    }
  }

  void _drawArm(
      Canvas canvas, double cx, double cy, double r, double side, double wave) {
    final shoulderX = cx + side * r * 0.77;
    final shoulderY = cy + r * 0.12;
    final elbowX = shoulderX + side * r * 0.28;
    final elbowY = shoulderY + r * 0.40 * math.cos(side * wave);
    final handX = elbowX + side * r * 0.16;
    final handY = elbowY + r * 0.28 - r * 0.3 * math.sin(side * wave + 0.3);

    final armPaint = Paint()
      ..color = const Color(0xFFD8E8F0)
      ..strokeWidth = r * 0.22
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final path = Path()
      ..moveTo(shoulderX, shoulderY)
      ..quadraticBezierTo(elbowX, elbowY, handX, handY);
    canvas.drawPath(path, armPaint);
    canvas.drawCircle(
      Offset(handX, handY),
      r * 0.14,
      Paint()..color = const Color(0xFFD8E8F0),
    );
  }

  @override
  bool shouldRepaint(covariant _RobotPainter oldDelegate) => true;
}
