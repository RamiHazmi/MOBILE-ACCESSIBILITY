import 'package:flutter/material.dart';
import 'package:local_auth/local_auth.dart';

import 'tts_service.dart';
import 'main_shell.dart';

class BiometricAuthScreen extends StatefulWidget {
  const BiometricAuthScreen({super.key});
  @override
  State<BiometricAuthScreen> createState() => _BiometricAuthScreenState();
}

class _BiometricAuthScreenState extends State<BiometricAuthScreen>
    with SingleTickerProviderStateMixin {
  final LocalAuthentication _auth = LocalAuthentication();

  bool _failed   = false;
  bool _checking = true;
  BiometricType? _biometricType; // detected biometric kind

  late AnimationController _pulseAnim;

  @override
  void initState() {
    super.initState();
    _pulseAnim = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1800))
      ..repeat(reverse: true);

    _authenticate();
  }

  @override
  void dispose() {
    _pulseAnim.dispose();
    super.dispose();
  }

  Future<void> _authenticate() async {
    if (!mounted) return;
    setState(() {
      _failed   = false;
      _checking = true;
    });

    await TtsService.init();

    try {
      final supported = await _auth.isDeviceSupported();
      print('[BiometricAuth] supported=$supported');

      if (!supported) {
        print('[BiometricAuth] Device has no auth support — skipping gate');
        _goHome();
        return;
      }

      await TtsService.speak('Please verify your identity.', lang: 'en');

      final ok = await _auth.authenticate(
        localizedReason: 'Look at the camera or use your PIN to log in',
        options: const AuthenticationOptions(
          biometricOnly: false, // allow weak face unlock + PIN fallback
          stickyAuth: true,
        ),
      );

      if (ok) {
        _goHome();
      } else {
        if (!mounted) return;
        setState(() {
          _failed   = true;
          _checking = false;
        });
        await TtsService.speak(
            'Not recognized. Tap anywhere to try again.', lang: 'en');
      }
    } catch (_) {
      // local_auth error (e.g. no fingerprints enrolled) → skip the gate
      _goHome();
    }
  }

  void _goHome() {
    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => const MainShell()),
    );
  }

  // ── UI ──────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: _failed ? _authenticate : null,
        child: Container(
          decoration: const BoxDecoration(
            gradient: RadialGradient(
              center: Alignment(0.0, -0.15),
              radius: 1.2,
              colors: [
                Color(0xFF0D2560),
                Color(0xFF060E22),
                Color(0xFF020508),
              ],
              stops: [0.0, 0.55, 1.0],
            ),
          ),
          child: SafeArea(
            child: Center(
              child: AnimatedBuilder(
                animation: _pulseAnim,
                builder: (_, __) {
                  final pulse = _pulseAnim.value;
                  final iconColor = _failed
                      ? Color.lerp(Colors.redAccent,
                          Colors.red.shade900, pulse)!
                      : Color.lerp(const Color(0xFF4FC3F7),
                          const Color(0xFF1565C0), pulse)!;

                  return Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      // Glow behind the icon
                      Container(
                        width: 140,
                        height: 140,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(
                              color: (_failed
                                      ? Colors.redAccent
                                      : const Color(0xFF4FC3F7))
                                  .withOpacity(0.20 + pulse * 0.18),
                              blurRadius: 48,
                              spreadRadius: 12,
                            ),
                          ],
                        ),
                        child: Icon(
                          _biometricType == BiometricType.face
                              ? Icons.face_unlock_outlined
                              : _biometricType == BiometricType.fingerprint
                                  ? Icons.fingerprint
                                  : Icons.lock_open_outlined,
                          size: 110,
                          color: iconColor,
                        ),
                      ),

                      const SizedBox(height: 40),

                      Text(
                        _failed
                            ? 'Not recognized\nTap to try again'
                            : 'Verifying identity…',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.72),
                          fontSize: 20,
                          fontWeight: FontWeight.w400,
                          height: 1.5,
                        ),
                      ),
                    ],
                  );
                },
              ),
            ),
          ),
        ),
      ),
    );
  }
}
