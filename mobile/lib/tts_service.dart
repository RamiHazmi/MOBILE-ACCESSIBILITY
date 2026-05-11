import 'package:flutter_tts/flutter_tts.dart';

/// On-device TTS via Android's free built-in engine.
/// 100% offline, 100% free.
///
/// Important: for Arabic / Tunisian Derja to actually be spoken,
/// the user must have the Arabic voice installed in Android:
///   Settings → System → Languages & input → Text-to-speech output
///   → Google TTS → Install voice data → Arabic (Saudi Arabia).
/// On Nokia 3.2 with Android 11 this is free and ~30 MB.
class TtsService {
  static final FlutterTts _tts = FlutterTts();
  static bool _initialized = false;
  // Cache which language codes are actually installed on the device
  static final Map<String, bool> _availableLangs = {};

  static Future<void> init() async {
    if (_initialized) return;
    await _tts.setSpeechRate(0.50);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
    await _tts.setQueueMode(1);          // 0 = flush, 1 = add
    await _tts.awaitSpeakCompletion(true);

    // Probe which locales the device supports — log so the user
    // can see what voices are missing and install them.
    try {
      const probes = ['en-US', 'fr-FR', 'ar-SA'];
      for (final loc in probes) {
        final ok = await _tts.isLanguageAvailable(loc);
        _availableLangs[loc] = ok == true;
      }
      print('🔊 TTS available voices: $_availableLangs');
      if (_availableLangs['ar-SA'] != true) {
        print('⚠️  Arabic voice not installed. '
              'Settings → Text-to-speech → Install voice data → Arabic.');
      }
    } catch (e) {
      print('🔊 TTS probe failed: $e');
    }

    await _tts.setLanguage('en-US');
    _initialized = true;
  }

  static Future<void> speak(
    String text, {
    String lang = 'en',
    bool   interrupt = false,
  }) async {
    if (text.trim().isEmpty) return;
    if (!_initialized) await init();

    const localeMap = {
      'en': 'en-US',
      'ar': 'ar-SA',
      'tn': 'ar-SA',     // Tunisian Derja → Arabic voice
      'fr': 'fr-FR',
    };
    String locale = localeMap[lang] ?? 'en-US';

    // If the requested locale isn't installed, fall back gracefully
    // to en-US so the user at least hears the message rather than
    // total silence.
    if (_availableLangs[locale] == false &&
        _availableLangs['en-US'] == true) {
      print('🔊 voice "$locale" not installed → falling back to en-US');
      locale = 'en-US';
    }

    await _tts.setLanguage(locale);

    if (interrupt) {
      await _tts.stop();
    }
    print('🔊 TTS ($lang/$locale): $text');
    await _tts.speak(text);
  }

  static Future<void> stop() async => _tts.stop();
}