import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';

import 'location_service.dart';

class ApiService {
  /// Change to the LAN IP of your dev machine running uvicorn / Flask.
  static const String baseUrl = 'http://192.168.1.13:8000';

  // ── Device UUID (generated once, stored locally) ─────────────
  static String? _cachedUserId;

  static Future<String> getUserId() async {
    if (_cachedUserId != null) return _cachedUserId!;
    final prefs = await SharedPreferences.getInstance();
    String? id = prefs.getString('wexist_user_id');
    if (id == null || id.isEmpty) {
      id = const Uuid().v4();
      await prefs.setString('wexist_user_id', id);
    }
    _cachedUserId = id;
    return id;
  }

  /// Returns headers including the X-User-ID for every request.
  static Future<Map<String, String>> _headers() async {
    final uid = await getUserId();
    return {'X-User-ID': uid};
  }



  // ── Voice / camera pipeline (existing) ──────────────────────

  static Future<String> process({
    required String audioPath,
    Uint8List? frameBytes,
  }) async {
    try {
      final audioFile = File(audioPath);
      if (!audioFile.existsSync()) {
        return '{"error":"audio file missing"}';
      }
      final audioSize = audioFile.lengthSync();
      print('📤 audio=$audioSize bytes  frame=${frameBytes?.length ?? 0}');
      if (audioSize < 64) return '{"error":"recording too short"}';

      final pos = await LocationService.current();
      if (pos != null) {
        print('📍 GPS: ${pos.latitude.toStringAsFixed(5)}, '
            '${pos.longitude.toStringAsFixed(5)}');
      } else {
        print('📍 GPS: unavailable');
      }

      final uri     = Uri.parse('$baseUrl/process');
      final request = http.MultipartRequest('POST', uri);

      // Attach user ID header
      final uid = await getUserId();
      request.headers['X-User-ID'] = uid;

      request.files.add(await http.MultipartFile.fromPath('audio', audioPath));

      if (frameBytes != null && frameBytes.isNotEmpty) {
        request.files.add(http.MultipartFile.fromBytes(
          'frame', frameBytes, filename: 'frame.jpg'));
      }
      if (pos != null) {
        request.fields['user_lat'] = pos.latitude.toString();
        request.fields['user_lng'] = pos.longitude.toString();
      }

      final streamed = await request.send()
          .timeout(const Duration(seconds: 120));
      final body = await streamed.stream.bytesToString();

      print('📡 ${streamed.statusCode}: ${body.length > 200
          ? "${body.substring(0, 200)}..."
          : body}');

      if (streamed.statusCode != 200) {
        return '{"error":"server ${streamed.statusCode}"}';
      }
      return body;
    } catch (e) {
      print('❌ API error: $e');
      return '{"error":"$e"}';
    }
  }

  // ── Handwriting OCR pipeline (new) ──────────────────────────

  /// POST /handwriting
  /// [imageB64]  base64-encoded JPEG or PNG of the handwritten image.
  /// [language]  one of: auto | english | french | arabic | darija.
  ///
  /// Returns the raw JSON string from the backend.
  /// On any network / timeout error returns '{"error":"..."}' so the
  /// caller never receives an exception — same pattern as process().
  static Future<String> handwriting({
    required String imageB64,
    String language = 'auto',
  }) async {
    try {
      final uri     = Uri.parse('$baseUrl/handwriting');
      final payload = jsonEncode({'image_b64': imageB64, 'language': language});

      print('✍️ POST /handwriting  lang=$language  '
          'payload=${(imageB64.length / 1024).toStringAsFixed(0)} KB');

      final hdrs = await _headers();
      hdrs['Content-Type'] = 'application/json';

      final response = await http
          .post(uri, headers: hdrs, body: payload)
          .timeout(const Duration(seconds: 120));

      print('📡 /handwriting ${response.statusCode}: '
          '${response.body.length > 200 ? "${response.body.substring(0, 200)}..." : response.body}');

      if (response.statusCode != 200) {
        return '{"error":"server ${response.statusCode}"}';
      }
      return response.body;
    } catch (e) {
      print('❌ handwriting API error: $e');
      return '{"error":"$e"}';
    }
  }

  // ── Personal person — save flow ─────────────────────────────

  /// POST /describe_for_save_person — VLM describes the face for confirmation.
  static Future<String> describeForSavePerson({
    required Uint8List frameBytes,
    required String    personName,
    String             relationship = 'person',
    String             language     = 'en',
  }) async {
    try {
      final uri     = Uri.parse('$baseUrl/describe_for_save_person');
      final request = http.MultipartRequest('POST', uri);
      final hdrs    = await _headers();
      request.headers.addAll(hdrs);
      request.fields['person_name']  = personName;
      request.fields['relationship'] = relationship;
      request.fields['language']     = language;
      request.files.add(http.MultipartFile.fromBytes(
          'frame', frameBytes, filename: 'frame.jpg'));
      final streamed = await request.send()
          .timeout(const Duration(seconds: 30));
      final body = await streamed.stream.bytesToString();
      if (streamed.statusCode != 200) return '{"error":"server ${streamed.statusCode}"}';
      return body;
    } catch (e) {
      print('❌ describeForSavePerson error: $e');
      return '{"error":"$e"}';
    }
  }

  /// POST /save_person — persist the confirmed person.
  static Future<String> savePersonRecord({
    required String name,
    required String relationship,
    required String description,
    String          imageB64 = '',
    String          language = 'en',
  }) async {
    try {
      final hdrs = await _headers();
      hdrs['Content-Type'] = 'application/json';
      final response = await http
          .post(
            Uri.parse('$baseUrl/save_person'),
            headers: hdrs,
            body: jsonEncode({
              'name':         name,
              'relationship': relationship,
              'description':  description,
              'image_b64':    imageB64,
              'language':     language,
            }),
          )
          .timeout(const Duration(seconds: 10));
      if (response.statusCode != 200) return '{"error":"server ${response.statusCode}"}';
      return response.body;
    } catch (e) {
      print('❌ savePersonRecord error: $e');
      return '{"error":"$e"}';
    }
  }

  static Future<bool> deletePerson(String name) async {
    try {
      final hdrs = await _headers();
      final response = await http
          .delete(
            Uri.parse('$baseUrl/user/persons/${Uri.encodeComponent(name)}'),
            headers: hdrs,
          )
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        return (jsonDecode(response.body) as Map<String, dynamic>)['ok'] == true;
      }
    } catch (_) {}
    return false;
  }

  // ── Personal object — save flow ─────────────────────────────

  /// POST /describe_for_save — VLM describes the frame for confirmation.
  static Future<String> describeForSave({
    required Uint8List frameBytes,
    required String    objectName,
    String             language = 'en',
  }) async {
    try {
      final uri     = Uri.parse('$baseUrl/describe_for_save');
      final request = http.MultipartRequest('POST', uri);

      final hdrs = await _headers();
      request.headers.addAll(hdrs);
      request.fields['name']     = objectName;
      request.fields['language'] = language;
      request.files.add(http.MultipartFile.fromBytes(
          'frame', frameBytes, filename: 'frame.jpg'));

      final streamed = await request.send()
          .timeout(const Duration(seconds: 30));
      final body = await streamed.stream.bytesToString();

      if (streamed.statusCode != 200) return '{"error":"server ${streamed.statusCode}"}';
      return body;
    } catch (e) {
      print('❌ describeForSave error: $e');
      return '{"error":"$e"}';
    }
  }

  /// POST /save_personal_object — persist the confirmed object.
  static Future<String> savePersonalObject({
    required String name,
    required String description,
    String          imageB64 = '',
    String          language = 'en',
  }) async {
    try {
      final hdrs = await _headers();
      hdrs['Content-Type'] = 'application/json';
      final response = await http
          .post(
            Uri.parse('$baseUrl/save_personal_object'),
            headers: hdrs,
            body: jsonEncode({
              'name':        name,
              'description': description,
              'image_b64':   imageB64,
              'language':    language,
            }),
          )
          .timeout(const Duration(seconds: 10));

      if (response.statusCode != 200) return '{"error":"server ${response.statusCode}"}';
      return response.body;
    } catch (e) {
      print('❌ savePersonalObject error: $e');
      return '{"error":"$e"}';
    }
  }

  // ── User profile ─────────────────────────────────────────────

  /// Fetches all profile data in one call: name, objects, locations, history.
  static Future<Map<String, dynamic>> getUserProfile() async {
    try {
      final hdrs = await _headers();
      final response = await http
          .get(Uri.parse('$baseUrl/user/profile'), headers: hdrs)
          .timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
    } catch (_) {}
    return {};
  }

  static Future<bool> deletePersonalObject(String name) async {
    try {
      final hdrs = await _headers();
      final response = await http
          .delete(
            Uri.parse('$baseUrl/user/objects/${Uri.encodeComponent(name)}'),
            headers: hdrs,
          )
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        return (jsonDecode(response.body) as Map<String, dynamic>)['ok'] == true;
      }
    } catch (_) {}
    return false;
  }

  static Future<bool> deleteNamedLocation(String label) async {
    try {
      final hdrs = await _headers();
      final response = await http
          .delete(
            Uri.parse('$baseUrl/user/locations/${Uri.encodeComponent(label)}'),
            headers: hdrs,
          )
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        return (jsonDecode(response.body) as Map<String, dynamic>)['ok'] == true;
      }
    } catch (_) {}
    return false;
  }

  static Future<bool> deleteHistoryEntry(int id) async {
    try {
      final hdrs = await _headers();
      final response = await http
          .delete(
            Uri.parse('$baseUrl/user/history/$id'),
            headers: hdrs,
          )
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        return (jsonDecode(response.body) as Map<String, dynamic>)['ok'] == true;
      }
    } catch (_) {}
    return false;
  }

  // ── Profile voice command ────────────────────────────────────

  /// POST /profile/command — transcribe + classify a voice command on
  /// the profile page using the user's actual DB data as context.
  static Future<Map<String, dynamic>> profileCommand({
    required String audioPath,
    String language      = 'en',
    String pendingAction = '',
  }) async {
    try {
      final uri     = Uri.parse('$baseUrl/profile/command');
      final request = http.MultipartRequest('POST', uri);
      final hdrs    = await _headers();
      request.headers.addAll(hdrs);
      request.fields['language']       = language;
      request.fields['pending_action'] = pendingAction;
      request.files.add(await http.MultipartFile.fromPath('audio', audioPath));
      final streamed = await request.send().timeout(const Duration(seconds: 45));
      final body     = await streamed.stream.bytesToString();
      if (streamed.statusCode != 200) {
        return {'error': 'server ${streamed.statusCode}'};
      }
      return jsonDecode(body) as Map<String, dynamic>;
    } catch (e) {
      print('❌ profileCommand error: $e');
      return {'error': '$e'};
    }
  }

  /// DELETE /user/history/object/{name} — remove all sightings for an object.
  static Future<bool> deleteHistoryByObject(String name) async {
    try {
      final hdrs = await _headers();
      final response = await http
          .delete(
            Uri.parse(
                '$baseUrl/user/history/object/${Uri.encodeComponent(name)}'),
            headers: hdrs,
          )
          .timeout(const Duration(seconds: 8));
      if (response.statusCode == 200) {
        return (jsonDecode(response.body) as Map<String, dynamic>)['ok'] == true;
      }
    } catch (_) {}
    return false;
  }

  // ── Text → Sign Language ────────────────────────────────────────

  /// POST /sign/text_to_sign
  /// Returns { tokens: [...], segments: [{token, type, video_url, letters:[...]}] }
  static Future<Map<String, dynamic>> textToSign({
    required String text,
    bool removeStopwords = false,
  }) async {
    try {
      final hdrs = await _headers();
      hdrs['Content-Type'] = 'application/json';
      final response = await http
          .post(
            Uri.parse('$baseUrl/sign/text_to_sign'),
            headers: hdrs,
            body: jsonEncode({
              'text': text,
              'remove_stopwords': removeStopwords,
            }),
          )
          .timeout(const Duration(seconds: 30));
      if (response.statusCode != 200) {
        return {'error': 'server ${response.statusCode}'};
      }
      return jsonDecode(response.body) as Map<String, dynamic>;
    } catch (e) {
      print('❌ textToSign error: $e');
      return {'error': '$e'};
    }
  }

  // ── VSR / lip reading ────────────────────────────────────────

  /// POST /vsr/translate — translate lip-reading output to french/arabic/darija.
  static Future<String> vsrTranslate({
    required String text,
    required String targetLanguage,
  }) async {
    try {
      final hdrs = await _headers();
      hdrs['Content-Type'] = 'application/json';
      final response = await http
          .post(
            Uri.parse('$baseUrl/vsr/translate'),
            headers: hdrs,
            body: jsonEncode({
              'text': text,
              'target_language': targetLanguage,
            }),
          )
          .timeout(const Duration(seconds: 30));
      if (response.statusCode != 200) return '';
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      return (data['translation'] as String? ?? '').trim();
    } catch (e) {
      print('❌ vsrTranslate error: $e');
      return '';
    }
  }

  // ── Reset (existing) ─────────────────────────────────────────

  /// Tell the backend to clear pending action / vision history.
  static Future<void> reset() async {
    try {
      final hdrs = await _headers();
      await http
          .post(Uri.parse('$baseUrl/reset'), headers: hdrs)
          .timeout(const Duration(seconds: 4));
    } catch (_) {}
  }

  // ── User profile ─────────────────────────────────────────────

  /// Returns the stored user name, or empty string if not set yet.
  static Future<String> getUserName() async {
    try {
      final hdrs = await _headers();
      final response = await http
          .get(Uri.parse('$baseUrl/user/name'), headers: hdrs)
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return (data['name'] as String? ?? '').trim();
      }
    } catch (_) {}
    return '';
  }

  /// Saves the user's name to the backend memory database.
  static Future<void> saveUserName(String name) async {
    try {
      final hdrs = await _headers();
      hdrs['Content-Type'] = 'application/json';
      await http
          .post(
            Uri.parse('$baseUrl/user/name'),
            headers: hdrs,
            body: jsonEncode({'name': name.trim()}),
          )
          .timeout(const Duration(seconds: 5));
    } catch (_) {}
  }
}