// mobile/lib/alarm_page.dart
// ══════════════════════════════════════════════════════════════
// Shows the history of detected environmental sounds.
// Opened when the user taps the danger badge on the home page.
// ══════════════════════════════════════════════════════════════

import 'package:flutter/material.dart';
import 'alarm_service.dart';

class AlarmPage extends StatefulWidget {
  const AlarmPage({super.key});
  @override
  State<AlarmPage> createState() => _AlarmPageState();
}

class _AlarmPageState extends State<AlarmPage> {
  @override
  void initState() {
    super.initState();
    // Dismiss the badge when user opens this page
    WidgetsBinding.instance.addPostFrameCallback((_) {
      AlarmService.instance.dismissAlert();
    });
    AlarmService.instance.history.addListener(_onHistoryUpdate);
  }

  @override
  void dispose() {
    AlarmService.instance.history.removeListener(_onHistoryUpdate);
    super.dispose();
  }

  void _onHistoryUpdate() {
    if (mounted) setState(() {});
  }

  // ── Category colours ─────────────────────────────────────────
  Color _catColor(String cat) {
    switch (cat) {
      case 'emergency': return const Color(0xFFFF3B30);
      case 'fire':      return const Color(0xFFFF6B35);
      case 'door':      return const Color(0xFF4C6FFF);
      case 'call':      return const Color(0xFF00AA55);
      case 'danger':    return const Color(0xFFFF3B30);
      case 'care':      return const Color(0xFFFF8A65);
      case 'alarm':     return const Color(0xFFFFCC00);
      case 'vehicle':   return const Color(0xFF4FC3F7);
      case 'animal':    return const Color(0xFF8BC34A);
      default:          return const Color(0xFF8BA0C8);
    }
  }

  Color _catBg(String cat) => _catColor(cat).withOpacity(0.12);

  @override
  Widget build(BuildContext context) {
    final events = AlarmService.instance.history.value;

    return Scaffold(
      backgroundColor: const Color(0xFF060E22),
      body: SafeArea(
        child: Column(children: [
          _buildHeader(context),
          Expanded(
            child: events.isEmpty
                ? _buildEmpty()
                : ListView.builder(
                    padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
                    itemCount: events.length,
                    itemBuilder: (_, i) => _AlarmCard(
                      event:    events[i],
                      catColor: _catColor(events[i].category),
                      catBg:    _catBg(events[i].category),
                      isLatest: i == 0,
                    ),
                  ),
          ),
          if (events.isNotEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: TextButton.icon(
                onPressed: () {
                  AlarmService.instance.history.value = [];
                  setState(() {});
                },
                icon: const Icon(Icons.delete_outline_rounded,
                    color: Color(0xFF8BA0C8), size: 18),
                label: const Text('Clear history',
                    style: TextStyle(color: Color(0xFF8BA0C8), fontSize: 13)),
              ),
            ),
        ]),
      ),
    );
  }

  Widget _buildHeader(BuildContext ctx) {
    final count = AlarmService.instance.history.value.length;
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFFFF3B30), Color(0xFFFF6B35)],
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
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('Sound Alerts',
                style: TextStyle(color: Colors.white,
                    fontSize: 20, fontWeight: FontWeight.w800)),
            Text('$count detection${count == 1 ? '' : 's'} recorded',
                style: const TextStyle(color: Color(0xFFFFE0DD), fontSize: 12)),
          ]),
        ),
        const Text('🔊', style: TextStyle(fontSize: 24)),
      ]),
    );
  }

  Widget _buildEmpty() => Center(
    child: Column(mainAxisSize: MainAxisSize.min, children: [
      const Text('🔇', style: TextStyle(fontSize: 64)),
      const SizedBox(height: 16),
      const Text('No sounds detected yet',
          style: TextStyle(color: Colors.white,
              fontSize: 18, fontWeight: FontWeight.w700)),
      const SizedBox(height: 8),
      Text('YAMNet monitors your environment\nand alerts you to important sounds.',
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 14, height: 1.5)),
    ]),
  );
}


// ── Individual alarm card ─────────────────────────────────────────
class _AlarmCard extends StatelessWidget {
  final AlarmEvent event;
  final Color      catColor;
  final Color      catBg;
  final bool       isLatest;
  const _AlarmCard({
    required this.event, required this.catColor,
    required this.catBg, required this.isLatest,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: catBg,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: isLatest ? catColor : catColor.withOpacity(0.3),
          width: isLatest ? 1.5 : 1,
        ),
        boxShadow: isLatest ? [
          BoxShadow(color: catColor.withOpacity(0.25), blurRadius: 12, offset: const Offset(0, 4)),
        ] : [],
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          // Emoji icon
          Container(
            width: 52, height: 52,
            decoration: BoxDecoration(
              color: catColor.withOpacity(0.15),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: catColor.withOpacity(0.4)),
            ),
            child: Center(child: Text(event.emoji,
                style: const TextStyle(fontSize: 26))),
          ),
          const SizedBox(width: 14),

          // Text content
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              if (isLatest)
                Container(
                  margin: const EdgeInsets.only(bottom: 5),
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: catColor,
                    borderRadius: BorderRadius.circular(99),
                  ),
                  child: const Text('LATEST',
                      style: TextStyle(color: Colors.white,
                          fontSize: 9, fontWeight: FontWeight.w800, letterSpacing: 1)),
                ),
              Text(event.description,
                  style: const TextStyle(color: Colors.white,
                      fontSize: 16, fontWeight: FontWeight.w700, height: 1.3)),
              const SizedBox(height: 4),
              Text(event.label,
                  style: TextStyle(color: catColor,
                      fontSize: 12, fontWeight: FontWeight.w600)),
              const SizedBox(height: 6),
              Row(children: [
                _ConfBar(confidence: event.confidence, color: catColor),
                const SizedBox(width: 8),
                Text('${(event.confidence * 100).toStringAsFixed(0)}%',
                    style: TextStyle(color: catColor,
                        fontSize: 11, fontWeight: FontWeight.w700)),
                const Spacer(),
                Icon(Icons.access_time_rounded,
                    color: Colors.white.withOpacity(0.4), size: 12),
                const SizedBox(width: 3),
                Text(event.ts,
                    style: TextStyle(color: Colors.white.withOpacity(0.45),
                        fontSize: 11)),
              ]),
            ]),
          ),
        ]),
      ),
    );
  }
}


// ── Confidence bar ────────────────────────────────────────────────
class _ConfBar extends StatelessWidget {
  final double confidence;
  final Color  color;
  const _ConfBar({required this.confidence, required this.color});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 60, height: 4,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(99),
        child: LinearProgressIndicator(
          value: confidence.clamp(0.0, 1.0),
          backgroundColor: Colors.white.withOpacity(0.1),
          valueColor: AlwaysStoppedAnimation<Color>(color),
        ),
      ),
    );
  }
}
