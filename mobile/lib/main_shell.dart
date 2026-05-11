import 'package:flutter/material.dart';
import 'ui_home.dart';
import 'profile_page.dart';
import 'tts_service.dart';

/// Root shell — horizontal PageView:
///   page 0 = HomePage  (swipe right → profile)
///   page 1 = ProfilePage (swipe left → home)
class MainShell extends StatefulWidget {
  const MainShell({super.key});
  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  final PageController _ctrl = PageController();

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  void _onPageChanged(int index) {
    if (index == 1) {
      TtsService.speak(
        "You're now in your profile. "
        "I can read your objects, locations, or history, "
        "or help you delete them. "
        "Tap the microphone and tell me what you need.",
        lang: 'en',
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return PageView(
      controller: _ctrl,
      onPageChanged: _onPageChanged,
      physics: const PageScrollPhysics(
        parent: ClampingScrollPhysics(),
      ),
      children: const [
        HomePage(),
        ProfilePage(),
      ],
    );
  }
}
