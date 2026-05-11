import 'dart:typed_data';
import 'package:camera/camera.dart';

/// Singleton camera service with a hard capture mutex.
/// A single [Completer]-style lock ensures takePicture() is never
/// called while the controller is being disposed.
class CameraService {
  CameraService._();

  static CameraController? _controller;
  static bool _initialized = false;
  static bool _disposing   = false;
  static bool _capturing   = false;

  // ── Lifecycle ────────────────────────────────────────────────

  static Future<void> init() async {
    if (_initialized || _disposing) return;

    List<CameraDescription> cameras;
    try {
      cameras = await availableCameras();
    } catch (e) {
      print('📷 availableCameras error: $e');
      return;
    }
    if (cameras.isEmpty) return;

    final ctrl = CameraController(
      cameras.first,
      ResolutionPreset.low,          // 240p – fast encode, short AE
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    try {
      await ctrl.initialize();
      // Disable flash BEFORE any capture so AE pre-capture never fires torch
      await ctrl.setFlashMode(FlashMode.off);
      // Lock exposure so the first frame isn't pitch-dark / overexposed
      await ctrl.setExposureMode(ExposureMode.auto);
      await ctrl.setFocusMode(FocusMode.auto);
    } catch (e) {
      print('📷 CameraService init error: $e');
      try { await ctrl.dispose(); } catch (_) {}
      return;
    }

    _controller  = ctrl;
    _initialized = true;
    _disposing   = false;
    print('📷 CameraService initialized');
  }

  // ── Capture ──────────────────────────────────────────────────

  /// Returns null if busy, not ready, or on any error.
  static Future<Uint8List?> captureFrame() async {
    if (!isReady || _capturing) return null;

    _capturing = true;
    try {
      final XFile file = await _controller!.takePicture();
      return await file.readAsBytes();
    } catch (e) {
      print('📷 Capture error: $e');
      return null;
    } finally {
      _capturing = false;
    }
  }

  // ── Dispose ──────────────────────────────────────────────────

  static Future<void> dispose() async {
    if (_disposing) return;
    _disposing   = true;
    _initialized = false;

    // Wait for any in-flight capture to finish (max 800 ms)
    int waited = 0;
    while (_capturing && waited < 800) {
      await Future<void>.delayed(const Duration(milliseconds: 50));
      waited += 50;
    }
    _capturing = false;

    final ctrl  = _controller;
    _controller = null;

    try {
      await ctrl?.dispose();
    } catch (_) {}

    _disposing = false;
    print('📷 CameraService disposed');
  }

  // ── State ────────────────────────────────────────────────────

  static bool get isReady =>
      _initialized &&
      !_disposing  &&
      !_capturing  &&
      _controller != null &&
      _controller!.value.isInitialized;
}