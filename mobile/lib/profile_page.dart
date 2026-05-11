import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'api_service.dart';
import 'recorder.dart';
import 'tts_service.dart';

// ── Voice state machine ───────────────────────────────────────
enum _VS { idle, recording, processing, confirming }

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage>
    with SingleTickerProviderStateMixin {
  // ── Colors ───────────────────────────────────────────────────
  static const Color _blue   = Color(0xFF4FC3F7);
  static const Color _cyan   = Color(0xFF00E5C0);
  static const Color _green  = Color(0xFF66BB6A);
  static const Color _orange = Color(0xFFFFA726);
  static const Color _peach  = Color(0xFFFF8A65);
  static const Color _bg     = Color(0xFF060E22);

  // ── Profile data ─────────────────────────────────────────────
  bool   _loading   = true;
  String _name      = '';
  List<Map<String, dynamic>> _persons   = [];
  List<Map<String, dynamic>> _objects   = [];
  List<Map<String, dynamic>> _locations = [];
  List<Map<String, dynamic>> _history   = [];

  // ── Voice state ───────────────────────────────────────────────
  _VS    _vs            = _VS.idle;
  String _language      = 'en';
  String _voiceStatus   = '';
  String _pendingAction = '';   // carries delete intent across turns ("which item?")
  Map<String, dynamic>? _pendingDelete;

  // ── Mic pulse animation ───────────────────────────────────────
  late AnimationController _pulseCtrl;
  StreamSubscription<double>? _ampSub;
  double _amplitude = 0.0;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);

    _ampSub = RecorderService.amplitudeStream.listen((v) {
      if (!mounted) return;
      setState(() => _amplitude = _amplitude + 0.3 * (v - _amplitude));
    });

    _load(welcome: true);
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    _ampSub?.cancel();
    super.dispose();
  }

  // ── Derived lists ─────────────────────────────────────────────

  // Person with relationship == 'self' (user's own saved face)
  Map<String, dynamic>? get _selfPerson {
    try {
      return _persons.firstWhere(
        (p) => (p['relationship'] as String? ?? '').toLowerCase() == 'self');
    } catch (_) { return null; }
  }

  // Friends/family only (excludes self)
  List<Map<String, dynamic>> get _friends => _persons
      .where((p) => (p['relationship'] as String? ?? '').toLowerCase() != 'self')
      .toList();

  // ── Data loading ──────────────────────────────────────────────

  bool _welcomeSpoken = false;

  Future<void> _load({bool welcome = false}) async {
    setState(() => _loading = true);
    final data = await ApiService.getUserProfile();
    if (!mounted) return;
    setState(() {
      _loading   = false;
      _name      = (data['name']    as String? ?? '').trim();
      _persons   = _toMapList(data['persons']);
      _objects   = _toMapList(data['objects']);
      _locations = _toMapList(data['locations']);
      _history   = _toMapList(data['history']);
    });
    if (welcome && !_welcomeSpoken) {
      _welcomeSpoken = true;
      _speakWelcome();
    }
  }

  void _speakWelcome() {
    final nm = _name.isNotEmpty ? _name : '';
    final fl = _friends.length;
    final ol = _objects.length;
    final ll = _locations.length;
    final hl = _history.length;
    final msg = {
      'en': '${nm.isNotEmpty ? "Welcome $nm. " : "Welcome. "}'
            'You have $fl contact${fl != 1 ? "s" : ""}, '
            '$ol object${ol != 1 ? "s" : ""}, '
            '$ll location${ll != 1 ? "s" : ""}, '
            'and $hl history entries. '
            'Tap anywhere to give a command.',
      'fr': '${nm.isNotEmpty ? "Bienvenue $nm. " : "Bienvenue. "}'
            'Vous avez $fl contact${fl != 1 ? "s" : ""}, '
            '$ol objet${ol != 1 ? "s" : ""}, '
            '$ll emplacement${ll != 1 ? "s" : ""} '
            'et $hl entrées d\'historique. '
            'Appuyez partout pour donner une commande.',
      'ar': '${nm.isNotEmpty ? "مرحباً $nm. " : "مرحباً. "}'
            'لديك $fl جهة اتصال، '
            '$ol غرض، '
            '$ll موقع، '
            'و$hl سجل. '
            'انقر في أي مكان لإعطاء أمر.',
      'tn': '${nm.isNotEmpty ? "مرحبا $nm. " : "مرحبا. "}'
            'عندك $fl شخص، '
            '$ol حاجة، '
            '$ll موقع، '
            'و$hl سجل. '
            'انقر في أي بلاصة باش تعطي أمر.',
    }[_language] ?? 'Welcome. Tap anywhere to give a command.';
    TtsService.speak(msg, lang: _language);
  }

  List<Map<String, dynamic>> _toMapList(dynamic raw) =>
      (raw as List? ?? [])
          .map((e) => Map<String, dynamic>.from(e as Map))
          .toList();

  // ── Delete helpers ────────────────────────────────────────────

  Future<void> _deletePerson(String name) async {
    final ok = await ApiService.deletePerson(name);
    if (!ok || !mounted) return;
    HapticFeedback.mediumImpact();
    setState(() => _persons.removeWhere((p) => p['name'] == name));
  }

  Future<void> _deleteObject(String name) async {
    final ok = await ApiService.deletePersonalObject(name);
    if (!ok || !mounted) return;
    HapticFeedback.mediumImpact();
    setState(() => _objects.removeWhere((o) => o['name'] == name));
  }

  Future<void> _deleteLocation(String label) async {
    final ok = await ApiService.deleteNamedLocation(label);
    if (!ok || !mounted) return;
    HapticFeedback.mediumImpact();
    setState(() => _locations.removeWhere((l) => l['label'] == label));
  }

  Future<void> _deleteHistory(int id) async {
    final ok = await ApiService.deleteHistoryEntry(id);
    if (!ok || !mounted) return;
    HapticFeedback.mediumImpact();
    setState(() =>
        _history.removeWhere((h) => (h['id'] as num?)?.toInt() == id));
  }

  // ── Tap-to-delete bottom sheet ────────────────────────────────

  void _confirmDelete(String title, String subtitle, VoidCallback onConfirm) {
    HapticFeedback.lightImpact();
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF0D1B3A),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 16, 24, 40),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Container(
            width: 40, height: 4,
            decoration: BoxDecoration(
              color: Colors.white24,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 22),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.redAccent.withOpacity(0.12),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.delete_outline_rounded,
                color: Colors.redAccent, size: 30),
          ),
          const SizedBox(height: 14),
          Text(title,
              style: const TextStyle(
                  color: Colors.white, fontSize: 17,
                  fontWeight: FontWeight.w700),
              textAlign: TextAlign.center),
          const SizedBox(height: 8),
          Text(subtitle,
              style: TextStyle(
                  color: Colors.white.withOpacity(0.5),
                  fontSize: 13, height: 1.4),
              textAlign: TextAlign.center),
          const SizedBox(height: 28),
          Row(children: [
            Expanded(
              child: _SheetBtn(
                label: 'Cancel',
                color: Colors.white24,
                textColor: Colors.white70,
                onTap: () => Navigator.pop(ctx),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _SheetBtn(
                label: 'Delete',
                color: Colors.redAccent.withOpacity(0.18),
                textColor: Colors.redAccent,
                border: Colors.redAccent.withOpacity(0.5),
                onTap: () { Navigator.pop(ctx); onConfirm(); },
              ),
            ),
          ]),
        ]),
      ),
    );
  }

  // ── Edit name ─────────────────────────────────────────────────

  Future<void> _editName() async {
    final ctrl   = TextEditingController(text: _name);
    final result = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF0D1B3A),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        title: const Text('Edit your name',
            style: TextStyle(color: Colors.white, fontSize: 17)),
        content: TextField(
          controller: ctrl,
          autofocus: true,
          style: const TextStyle(color: Colors.white, fontSize: 16),
          decoration: InputDecoration(
            hintText: 'Your name',
            hintStyle: const TextStyle(color: Colors.white38),
            enabledBorder: UnderlineInputBorder(
                borderSide: BorderSide(color: _blue.withOpacity(0.4))),
            focusedBorder: const UnderlineInputBorder(
                borderSide: BorderSide(color: _blue)),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel',
                style: TextStyle(color: Colors.white54)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, ctrl.text.trim()),
            child: const Text('Save',
                style: TextStyle(color: _blue, fontWeight: FontWeight.w700)),
          ),
        ],
      ),
    );
    if (result != null && result.isNotEmpty && result != _name) {
      await ApiService.saveUserName(result);
      if (!mounted) return;
      HapticFeedback.lightImpact();
      setState(() => _name = result);
    }
  }

  // ── Voice: record ─────────────────────────────────────────────

  Future<void> _startVoice() async {
    final path = await RecorderService.startRecording();
    if (path == null) {
      setState(() => _voiceStatus = 'Microphone permission denied');
      return;
    }
    HapticFeedback.lightImpact();
    setState(() {
      _vs          = _VS.recording;
      _voiceStatus = 'Listening… tap to stop';
      _amplitude   = 0;
    });
  }

  Future<void> _stopVoiceAndProcess() async {
    setState(() {
      _vs          = _VS.processing;
      _voiceStatus = 'Thinking…';
      _amplitude   = 0;
    });
    final audioPath = await RecorderService.stopRecording();
    if (audioPath == null) {
      setState(() { _vs = _VS.idle; _voiceStatus = ''; });
      return;
    }
    final data = await ApiService.profileCommand(
        audioPath: audioPath, language: _language,
        pendingAction: _pendingAction);
    if (!mounted) return;
    _handleProfileCommand(data);
  }

  void _handleProfileCommand(Map<String, dynamic> data) {
    if (data.containsKey('error')) {
      setState(() { _vs = _VS.idle; _voiceStatus = ''; _pendingAction = ''; });
      TtsService.speak('Something went wrong. Please try again.',
          lang: _language);
      return;
    }

    final action         = (data['action']           as String? ?? 'unknown');
    final speak          = (data['speak']            as String? ?? '');
    final confirm        = (data['confirm_required'] as bool?   ?? false);
    final awaitTarget    = (data['awaiting_target']  as bool?   ?? false);
    final tName          = (data['target_name']      as String?);
    final tIds           = (data['target_ids'] as List?)
                              ?.map((e) => (e as num).toInt()).toList() ?? <int>[];
    final detLang        = (data['language']         as String? ?? _language);

    setState(() => _language = detLang);

    if (speak.isNotEmpty) {
      final _gesture = (confirm && tName != null)
          ? const {
              'en': ' Tap once to confirm, double tap to cancel.',
              'fr': ' Appuyez une fois pour confirmer, double appui pour annuler.',
              'ar': ' انقر مرة للتأكيد، انقر مرتين للإلغاء.',
              'tn': ' انقر مرة باش تأكد، انقر مرتين باش تلغي.',
            }
          : null;
      final hint = _gesture?[detLang] ?? (_gesture?['en'] ?? '');
      TtsService.speak('$speak$hint', lang: detLang);
    }

    // Rename was already executed server-side — just refresh the page data.
    if (action.startsWith('rename_')) {
      setState(() { _vs = _VS.idle; _voiceStatus = ''; _pendingAction = ''; });
      _load();
      return;
    }

    if (confirm && tName != null) {
      // Backend found the target in DB — show YES/NO overlay
      HapticFeedback.mediumImpact();
      setState(() {
        _vs            = _VS.confirming;
        _voiceStatus   = '';
        _pendingAction = '';
        _pendingDelete = {
          'action':      action,
          'target_name': tName,
          'target_ids':  tIds,
        };
      });
    } else if (awaitTarget) {
      // Backend knows the intent (delete_*) but needs the item name —
      // store pending_action so the NEXT tap sends it as context.
      final hint = {
        'en': 'Say the name of the item…',
        'fr': 'Dites le nom de l\'élément…',
        'ar': 'قل اسم العنصر…',
        'tn': 'قول الاسم…',
      }[detLang] ?? 'Say the name…';
      setState(() {
        _vs            = _VS.idle;
        _voiceStatus   = hint;
        _pendingAction = action;   // remembered for next tap
      });
    } else {
      setState(() { _vs = _VS.idle; _voiceStatus = ''; _pendingAction = ''; });
    }
  }

  // ── Voice: YES / NO ───────────────────────────────────────────

  void _cancelConfirm() {
    HapticFeedback.lightImpact();
    setState(() {
      _vs = _VS.idle; _pendingDelete = null;
      _voiceStatus = ''; _pendingAction = '';
    });
    TtsService.speak(
      {'en': 'Cancelled.', 'fr': 'Annulé.', 'ar': 'تم الإلغاء.', 'tn': 'لغينا.'}[_language]
          ?? 'Cancelled.',
      lang: _language,
    );
  }

  Future<void> _executeConfirmedDelete() async {
    final p = _pendingDelete;
    if (p == null) return;
    setState(() { _vs = _VS.idle; _pendingDelete = null; _voiceStatus = ''; });
    HapticFeedback.mediumImpact();

    final action = p['action'] as String? ?? '';
    final tName  = p['target_name'] as String? ?? '';
    final tIds   = (p['target_ids'] as List?)
                      ?.map((e) => (e as num).toInt()).toList() ?? <int>[];

    bool ok = false;
    if (action == 'delete_person') {
      ok = await ApiService.deletePerson(tName);
      if (ok && mounted) {
        setState(() => _persons.removeWhere((p) => p['name'] == tName));
      }
    } else if (action == 'delete_object') {
      ok = await ApiService.deletePersonalObject(tName);
      if (ok && mounted) {
        setState(() => _objects.removeWhere((o) => o['name'] == tName));
      }
    } else if (action == 'delete_location') {
      ok = await ApiService.deleteNamedLocation(tName);
      if (ok && mounted) {
        setState(() => _locations.removeWhere((l) => l['label'] == tName));
      }
    } else if (action == 'delete_history') {
      // Delete all sightings for that object in one backend call
      ok = await ApiService.deleteHistoryByObject(tName);
      if (ok && mounted) {
        setState(() =>
            _history.removeWhere((h) => h['object_name'] == tName));
      }
      // Also cover individual IDs passed from backend
      if (!ok && tIds.isNotEmpty) {
        for (final id in tIds) {
          await ApiService.deleteHistoryEntry(id);
        }
        if (mounted) {
          setState(() => _history.removeWhere(
              (h) => tIds.contains((h['id'] as num?)?.toInt())));
          ok = true;
        }
      }
    }

    if (!mounted) return;
    TtsService.speak(
      ok
          ? {'en': 'Deleted successfully.',
             'fr': 'Supprimé avec succès.',
             'ar': 'تم الحذف بنجاح.',
             'tn': 'تحذف بنجاح.'}[_language] ?? 'Deleted.'
          : {'en': 'Could not delete. Please try again.',
             'fr': 'Impossible de supprimer. Réessayez.',
             'ar': 'لم أتمكن من الحذف. حاول مرة أخرى.',
             'tn': 'ما نجمتش تمسح. عاود.'}[_language] ?? 'Could not delete.',
      lang: _language,
    );
  }

  // ── Build ─────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      body: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () {
          if (_vs == _VS.confirming) return;
          if (_vs == _VS.recording) {
            _stopVoiceAndProcess();
          } else if (_vs == _VS.idle) {
            _startVoice();
          }
        },
        onDoubleTap: () {
          if (_vs == _VS.confirming) _cancelConfirm();
        },
        child: Stack(fit: StackFit.expand, children: [
        // Background gradient
        const DecoratedBox(
          decoration: BoxDecoration(
            gradient: RadialGradient(
              center: Alignment(0.0, -0.35),
              radius: 1.15,
              colors: [
                Color(0xFF0A1A3A),
                Color(0xFF060E22),
                Color(0xFF020508),
              ],
              stops: [0.0, 0.5, 1.0],
            ),
          ),
        ),

        SafeArea(
          child: _loading
              ? const Center(
                  child: CircularProgressIndicator(
                      color: _blue, strokeWidth: 2))
              : RefreshIndicator(
                  onRefresh: _load,
                  color: _blue,
                  backgroundColor: const Color(0xFF0D1B3A),
                  child: CustomScrollView(
                    physics: const AlwaysScrollableScrollPhysics(),
                    slivers: [
                      SliverToBoxAdapter(child: _buildHeader()),

                      // ── Personal People ───────────────────────
                      SliverToBoxAdapter(
                        child: _SectionHeader(
                          icon: Icons.people_rounded,
                          color: _peach,
                          title: 'Personal People',
                          count: _friends.length,
                        ),
                      ),
                      _friends.isEmpty
                          ? SliverToBoxAdapter(
                              child: _EmptyHint(
                                'No saved people yet.\nSay "this is my friend [name]" to add someone.',
                              ),
                            )
                          : SliverList(
                              delegate: SliverChildBuilderDelegate(
                                (_, i) => _buildPersonTile(_friends[i]),
                                childCount: _friends.length,
                              ),
                            ),

                      // ── Personal Objects ──────────────────────
                      SliverToBoxAdapter(
                        child: _SectionHeader(
                          icon: Icons.bookmark_rounded,
                          color: _cyan,
                          title: 'Personal Objects',
                          count: _objects.length,
                        ),
                      ),
                      _objects.isEmpty
                          ? SliverToBoxAdapter(
                              child: _EmptyHint(
                                'No saved objects yet.\nSay "save this as [name]" to add one.',
                              ),
                            )
                          : SliverList(
                              delegate: SliverChildBuilderDelegate(
                                (_, i) => _buildObjectTile(_objects[i]),
                                childCount: _objects.length,
                              ),
                            ),

                      // ── Saved Locations ───────────────────────
                      SliverToBoxAdapter(
                        child: _SectionHeader(
                          icon: Icons.place_rounded,
                          color: _green,
                          title: 'Saved Locations',
                          count: _locations.length,
                        ),
                      ),
                      _locations.isEmpty
                          ? SliverToBoxAdapter(
                              child: _EmptyHint(
                                'No saved locations.\nSay "save this location as home" to add one.',
                              ),
                            )
                          : SliverList(
                              delegate: SliverChildBuilderDelegate(
                                (_, i) => _buildLocationTile(_locations[i]),
                                childCount: _locations.length,
                              ),
                            ),

                      // ── Object History ────────────────────────
                      SliverToBoxAdapter(
                        child: _SectionHeader(
                          icon: Icons.history_rounded,
                          color: _orange,
                          title: 'Object History',
                          count: _history.length,
                        ),
                      ),
                      _history.isEmpty
                          ? SliverToBoxAdapter(
                              child: _EmptyHint(
                                'No history yet.\nHistory builds as you find objects.',
                              ),
                            )
                          : SliverList(
                              delegate: SliverChildBuilderDelegate(
                                (_, i) => _buildHistoryTile(_history[i]),
                                childCount: _history.length,
                              ),
                            ),

                      // Bottom padding so content clears the mic bar
                      const SliverToBoxAdapter(child: SizedBox(height: 80)),
                    ],
                  ),
                ),
        ),

        // ── Floating mic bar ──────────────────────────────────
        if (_vs != _VS.confirming)
          Positioned(
            bottom: 0, left: 0, right: 0,
            child: SafeArea(top: false, child: _buildMicBar()),
          ),

        // ── Confirmation overlay ──────────────────────────────
        if (_vs == _VS.confirming && _pendingDelete != null)
          _buildConfirmOverlay(),
      ]),
      ),
    );
  }

  // ── Header card ───────────────────────────────────────────────

  Widget _buildHeader() => Container(
        margin: const EdgeInsets.fromLTRB(16, 10, 16, 4),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.04),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: _blue.withOpacity(0.18), width: 1),
        ),
        child: Column(children: [
          Row(children: [
            Row(children: [
              Icon(Icons.chevron_left_rounded,
                  color: Colors.white.withOpacity(0.22), size: 16),
              Text('swipe',
                  style: TextStyle(
                      color: Colors.white.withOpacity(0.22),
                      fontSize: 10,
                      letterSpacing: 0.3)),
            ]),
            const Spacer(),
            Text('My Profile',
                style: TextStyle(
                    color: Colors.white.withOpacity(0.55),
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.5)),
            const Spacer(),
            GestureDetector(
              onTap: _load,
              child: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.07),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(Icons.refresh_rounded,
                    color: Colors.white.withOpacity(0.45), size: 15),
              ),
            ),
          ]),
          const SizedBox(height: 20),
          Container(
            width: 76, height: 76,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(colors: [
                _blue.withOpacity(0.22),
                const Color(0xFF0A1A3A),
              ]),
              border: Border.all(color: _blue.withOpacity(0.4), width: 1.5),
            ),
            child: Builder(builder: (_) {
              final imgB64 = (_selfPerson?['image_b64'] as String? ?? '').trim();
              if (imgB64.isNotEmpty) {
                try {
                  final bytes = base64Decode(imgB64);
                  return ClipOval(
                    child: Image.memory(
                        bytes, width: 76, height: 76, fit: BoxFit.cover),
                  );
                } catch (_) {}
              }
              return const Icon(Icons.person_rounded, color: _blue, size: 38);
            }),
          ),
          const SizedBox(height: 14),
          GestureDetector(
            onTap: _editName,
            child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              Text(
                _name.isEmpty ? 'Tap to set your name' : _name,
                style: TextStyle(
                  color: _name.isEmpty ? Colors.white38 : Colors.white,
                  fontSize: 21,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(width: 8),
              Icon(Icons.edit_rounded, color: _blue.withOpacity(0.6), size: 15),
            ]),
          ),
          const SizedBox(height: 8),
          Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            _Stat(value: _friends.length,   label: 'People',    color: _peach),
            _Divider(),
            _Stat(value: _objects.length,   label: 'Objects',   color: _cyan),
            _Divider(),
            _Stat(value: _locations.length, label: 'Locations', color: _green),
            _Divider(),
            _Stat(value: _history.length,   label: 'Sightings', color: _orange),
          ]),
        ]),
      );

  // ── Tile builders ─────────────────────────────────────────────

  Widget _buildPersonTile(Map<String, dynamic> p) {
    final name    = (p['name']             as String? ?? '').trim();
    final rel     = (p['relationship']     as String? ?? '').trim();
    final desc    = (p['face_description'] as String? ?? '').trim();
    final imgB64  = (p['image_b64']        as String? ?? '').trim();

    // Build circular avatar: photo if available, icon fallback
    Widget avatar;
    if (imgB64.isNotEmpty) {
      try {
        final bytes = base64Decode(imgB64);
        avatar = ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: Image.memory(bytes, width: 40, height: 40, fit: BoxFit.cover),
        );
      } catch (_) {
        avatar = _personIconBox();
      }
    } else {
      avatar = _personIconBox();
    }

    return Container(
      margin:  const EdgeInsets.fromLTRB(16, 0, 16, 8),
      padding: const EdgeInsets.fromLTRB(14, 13, 14, 13),
      decoration: BoxDecoration(
        color:        Colors.white.withOpacity(0.03),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
            color: Colors.white.withOpacity(0.07), width: 1),
      ),
      child: Row(children: [
        avatar,
        const SizedBox(width: 13),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(name,
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 15,
                      fontWeight: FontWeight.w600)),
              const SizedBox(height: 3),
              Text(
                rel.isNotEmpty
                    ? '$rel${desc.isNotEmpty ? " · ${desc.substring(0, desc.length.clamp(0, 60))}" : ""}'
                    : (desc.isNotEmpty
                        ? desc.substring(0, desc.length.clamp(0, 60))
                        : 'No description'),
                style: TextStyle(
                    color: Colors.white.withOpacity(0.42),
                    fontSize: 12, height: 1.4),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),
      ]),
    );
  }

  Widget _personIconBox() => Container(
    width: 40, height: 40,
    decoration: BoxDecoration(
      color: _peach.withOpacity(0.11),
      borderRadius: BorderRadius.circular(20),
    ),
    child: const Icon(Icons.person_rounded, color: _peach, size: 20),
  );

  Widget _buildObjectTile(Map<String, dynamic> obj) {
    final name = (obj['name'] as String? ?? '').trim();
    final desc = (obj['description'] as String? ?? '').trim();
    return _InfoTile(
      icon: Icons.bookmark_rounded, iconColor: _cyan,
      title: name,
      subtitle: desc.isNotEmpty ? desc : 'No description',
    );
  }

  Widget _buildLocationTile(Map<String, dynamic> loc) {
    final label = (loc['label']   as String? ?? '').trim();
    final lat   = (loc['gps_lat'] as num?)?.toStringAsFixed(4) ?? '?';
    final lng   = (loc['gps_lng'] as num?)?.toStringAsFixed(4) ?? '?';
    return _InfoTile(
      icon: Icons.place_rounded, iconColor: _green,
      title: label,
      subtitle: '$lat, $lng',
    );
  }

  Widget _buildHistoryTile(Map<String, dynamic> h) {
    final name    = (h['object_name'] as String? ?? '').trim();
    final timeAgo = (h['time_ago'] as String? ?? '').trim();
    final scene   = (h['scene_description'] as String? ?? '').trim();
    final sub = [if (timeAgo.isNotEmpty) timeAgo,
                 if (scene.isNotEmpty) scene].join(' · ');
    return _InfoTile(
      icon: Icons.history_rounded, iconColor: _orange,
      title: name,
      subtitle: sub.isNotEmpty ? sub : 'No details',
    );
  }

  // ── Mic bar ───────────────────────────────────────────────────

  Widget _buildMicBar() {
    final isRecording  = _vs == _VS.recording;
    final isProcessing = _vs == _VS.processing;

    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter,
          end: Alignment.topCenter,
          colors: [
            const Color(0xFF060E22).withOpacity(0.98),
            const Color(0xFF060E22).withOpacity(0.0),
          ],
        ),
      ),
      padding: const EdgeInsets.fromLTRB(0, 14, 0, 28),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        // Status / hint text
        AnimatedSwitcher(
          duration: const Duration(milliseconds: 200),
          child: _voiceStatus.isNotEmpty
              ? Padding(
                  key: ValueKey(_voiceStatus),
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Text(
                    _voiceStatus,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: isRecording
                          ? Colors.redAccent.withOpacity(0.8)
                          : Colors.white38,
                      fontSize: 12,
                    ),
                  ),
                )
              : Padding(
                  key: const ValueKey('hint'),
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Text(
                    'tap anywhere to talk',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.18),
                      fontSize: 11,
                    ),
                  ),
                ),
        ),

        // Wave while recording; spinner while processing
        if (isRecording)
          SizedBox(
            height: 32,
            child: _MicWave(amplitude: _amplitude),
          )
        else if (isProcessing)
          const SizedBox(
            width: 24, height: 24,
            child: CircularProgressIndicator(color: _blue, strokeWidth: 2),
          ),
      ]),
    );
  }

  // ── Confirmation overlay ──────────────────────────────────────

  Widget _buildConfirmOverlay() {
    final p      = _pendingDelete!;
    final action = p['action']      as String? ?? '';
    final tName  = p['target_name'] as String? ?? '';
    final L      = _language;

    final typeLabel = <String, Map<String, String>>{
      'delete_person':   {'en': 'Person',   'fr': 'Personne',    'ar': 'شخص',  'tn': 'شخص'},
      'delete_object':   {'en': 'Object',   'fr': 'Objet',       'ar': 'غرض',  'tn': 'حاجة'},
      'delete_location': {'en': 'Location', 'fr': 'Emplacement', 'ar': 'موقع', 'tn': 'موقع'},
      'delete_history':  {'en': 'History',  'fr': 'Historique',  'ar': 'سجل',  'tn': 'سجل'},
    }[action]?[L] ?? 'Delete';

    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap:       _executeConfirmedDelete,
      onDoubleTap: _cancelConfirm,
      child: AnimatedOpacity(
        opacity: 1.0,
        duration: const Duration(milliseconds: 180),
        child: Container(
          color: Colors.black.withOpacity(0.90),
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 28),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.redAccent.withOpacity(0.12),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(Icons.delete_rounded,
                        color: Colors.redAccent, size: 44),
                  ),
                  const SizedBox(height: 24),
                  Text(
                    'Delete $typeLabel',
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 14),
                  Text(
                    '"$tName"',
                    style: const TextStyle(
                        color: Colors.redAccent,
                        fontSize: 28,
                        fontWeight: FontWeight.w800),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 10),
                  Text(
                    {'en': 'This cannot be undone.',
                     'fr': 'Cela ne peut pas être annulé.',
                     'ar': 'لا يمكن التراجع عن هذا.',
                     'tn': 'ما تنجمش ترجع لورا.'}[L] ?? 'This cannot be undone.',
                    style: TextStyle(
                        color: Colors.white.withOpacity(0.4), fontSize: 13),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 44),
                  Text(
                    {'en': '• tap once to confirm  •• double tap to cancel',
                     'fr': '• appui simple pour confirmer  •• double appui pour annuler',
                     'ar': '• انقر مرة للتأكيد  •• انقر مرتين للإلغاء',
                     'tn': '• انقر مرة باش تأكد  •• انقر مرتين باش تلغي'}[L]
                        ?? '• tap once to confirm  •• double tap to cancel',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.35),
                      fontSize: 13,
                      height: 1.5,
                      letterSpacing: 0.2,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ─── Mic wave painter ──────────────────────────────────────────

class _MicWave extends StatefulWidget {
  final double amplitude;
  const _MicWave({required this.amplitude});
  @override
  State<_MicWave> createState() => _MicWaveState();
}

class _MicWaveState extends State<_MicWave>
    with SingleTickerProviderStateMixin {
  late AnimationController _t;
  @override
  void initState() {
    super.initState();
    _t = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 800))
      ..repeat();
  }
  @override
  void dispose() { _t.dispose(); super.dispose(); }
  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _t,
      builder: (_, __) => CustomPaint(
        painter: _WavePainter(_t.value, widget.amplitude),
        size: const Size(double.infinity, 32),
      ),
    );
  }
}

class _WavePainter extends CustomPainter {
  final double t;
  final double amp;
  const _WavePainter(this.t, this.amp);

  @override
  void paint(Canvas canvas, Size size) {
    final midY = size.height / 2;
    final a    = size.height * 0.10 + amp * size.height * 0.36;
    final path = Path();
    for (int i = 0; i <= 200; i++) {
      final x = size.width * i / 200;
      final y = midY +
          a * math.sin(2.0 * math.pi * (i / 200) * 1.6 + t * math.pi * 2 * 2.5);
      if (i == 0) path.moveTo(x, y); else path.lineTo(x, y);
    }
    canvas.drawPath(
      path,
      Paint()
        ..color      = Colors.redAccent.withOpacity(0.8)
        ..style      = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..strokeCap  = StrokeCap.round,
    );
  }

  @override
  bool shouldRepaint(_WavePainter old) => old.t != t || old.amp != amp;
}

// ─── Shared small widgets ──────────────────────────────────────

class _SectionHeader extends StatelessWidget {
  final IconData icon;
  final Color    color;
  final String   title;
  final int      count;
  const _SectionHeader(
      {required this.icon, required this.color,
       required this.title, required this.count});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.fromLTRB(20, 26, 20, 10),
        child: Row(children: [
          Icon(icon, color: color, size: 17),
          const SizedBox(width: 9),
          Text(title.toUpperCase(),
              style: TextStyle(
                  color: color, fontSize: 11,
                  fontWeight: FontWeight.w800, letterSpacing: 1.1)),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(
              color: color.withOpacity(0.12),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text('$count',
                style: TextStyle(
                    color: color, fontSize: 11, fontWeight: FontWeight.w700)),
          ),
        ]),
      );
}

class _InfoTile extends StatelessWidget {
  final IconData  icon;
  final Color     iconColor;
  final String    title;
  final String    subtitle;
  const _InfoTile(
      {required this.icon, required this.iconColor,
       required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) => Container(
        margin: const EdgeInsets.fromLTRB(16, 0, 16, 8),
        padding: const EdgeInsets.fromLTRB(14, 13, 14, 13),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.03),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
              color: Colors.white.withOpacity(0.07), width: 1),
        ),
        child: Row(children: [
          Container(
            width: 40, height: 40,
            decoration: BoxDecoration(
              color: iconColor.withOpacity(0.11),
              borderRadius: BorderRadius.circular(11),
            ),
            child: Icon(icon, color: iconColor, size: 20),
          ),
          const SizedBox(width: 13),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 15,
                        fontWeight: FontWeight.w600)),
                const SizedBox(height: 3),
                Text(subtitle,
                    style: TextStyle(
                        color: Colors.white.withOpacity(0.42),
                        fontSize: 12, height: 1.4),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis),
              ],
            ),
          ),
        ]),
      );
}

class _SheetBtn extends StatelessWidget {
  final String       label;
  final Color        color;
  final Color        textColor;
  final Color?       border;
  final VoidCallback onTap;
  const _SheetBtn(
      {required this.label, required this.color,
       required this.textColor, required this.onTap, this.border});

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(13),
            border: border != null
                ? Border.all(color: border!, width: 1)
                : null,
          ),
          child: Text(label,
              textAlign: TextAlign.center,
              style: TextStyle(
                  color: textColor,
                  fontSize: 15,
                  fontWeight: FontWeight.w700)),
        ),
      );
}

class _Stat extends StatelessWidget {
  final int    value;
  final String label;
  final Color  color;
  const _Stat({required this.value, required this.label, required this.color});

  @override
  Widget build(BuildContext context) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('$value',
              style: TextStyle(
                  color: color, fontSize: 20, fontWeight: FontWeight.w800)),
          const SizedBox(height: 2),
          Text(label,
              style: TextStyle(
                  color: Colors.white.withOpacity(0.38),
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.4)),
        ],
      );
}

class _Divider extends StatelessWidget {
  @override
  Widget build(BuildContext context) => Container(
        height: 28, width: 1,
        margin: const EdgeInsets.symmetric(horizontal: 20),
        color: Colors.white.withOpacity(0.1),
      );
}

class _EmptyHint extends StatelessWidget {
  final String message;
  const _EmptyHint(this.message);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.fromLTRB(32, 4, 32, 12),
        child: Text(message,
            textAlign: TextAlign.center,
            style: TextStyle(
                color: Colors.white.withOpacity(0.25),
                fontSize: 13, height: 1.55)),
      );
}
