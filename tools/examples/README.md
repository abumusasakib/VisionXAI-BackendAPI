# Flutter: Decode & Render Attention Overlay (example)

This lightweight example shows how a Flutter client can:

- Decode the server's `attention_image` (base64 PNG) into bytes and display it.
- Parse `attention_color_map` (preferred) or `attention_colors` + `tokens` fallback.
- Render token markers at `attention_topk_items` locations using a `CustomPainter` overlay.

## Notes

- This example is intentionally minimal — adapt to your app's state management and UI patterns.
- `attention_grid` (rows, cols) is required to map grid coordinates -> pixel positions.

```dart
// Minimal imports
import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

// 1) Decode base64 PNG into a Flutter Image (ui.Image) for a CustomPainter
Future<ui.Image> decodeImageFromBase64(String b64) async {
  final bytes = base64Decode(b64);
  final completer = Completer<ui.Image>();
  ui.decodeImageFromList(bytes, (img) => completer.complete(img));
  return completer.future;
}

// 2) Parse colors: prefer attention_color_map, else attention_colors + tokens
Map<String, Color> parseTokenColorMap(Map<String, dynamic>? colorMap, List<String>? tokens, List<String>? colors) {
  Map<String, Color> out = {};
  Color parseHex(String hex) {
    hex = hex.replaceFirst('#', '');
    return Color(int.parse('0xff' + hex));
  }

  if (colorMap != null) {
    colorMap.forEach((k, v) {
      try { out[k] = parseHex(v as String); } catch (_) {}
    });
    return out;
  }

  if (colors != null && tokens != null) {
    for (var i = 0; i < tokens.length && i < colors.length; i++) {
      try { out[tokens[i]] = parseHex(colors[i]); } catch (_) {}
    }
  }
  return out;
}

// 3) CustomPainter: draw background image and marker circles for each token top-k
class TokenMarkerPainter extends CustomPainter {
  final ui.Image backgroundImage;
  final List<List<Map<String, dynamic>>> topkItems;
  final List<String> tokens;
  final Map<String, Color> tokenColors;
  final int rows;
  final int cols;

  TokenMarkerPainter({required this.backgroundImage, required this.topkItems, required this.tokens, required this.tokenColors, required this.rows, required this.cols});

  @override
  void paint(Canvas canvas, Size size) {
    // Paint full image
    paintImage(canvas: canvas, rect: Offset.zero & size, image: backgroundImage, fit: BoxFit.fill);

    final cellW = size.width / cols;
    final cellH = size.height / rows;
    final paint = Paint()..style = PaintingStyle.fill;

    for (int i = 0; i < topkItems.length && i < tokens.length; i++) {
      final token = tokens[i];
      final color = tokenColors[token] ?? Colors.red;
      paint.color = color.withOpacity(0.9);
      for (var entry in topkItems[i]) {
        final r = (entry['row'] as num).toDouble();
        final c = (entry['col'] as num).toDouble();
        final cx = (c + 0.5) * cellW;
        final cy = (r + 0.5) * cellH;
        canvas.drawCircle(Offset(cx, cy), (cellW < cellH ? cellW : cellH) * 0.12, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

// 4) Usage sketch (inside a StatefulWidget):
// - Fetch response JSON from server
// - Decode image, parse color map, then rebuild a Stack with the rendered image and CustomPaint overlay

/*
final response = await http.post(...); // POST /caption
final Map<String,dynamic> r = jsonDecode(response.body);
final String? b64 = r['attention_image'] as String?;
final ui.Image bg = await decodeImageFromBase64(b64!);
final tokens = (r['tokens'] as List?)?.cast<String>();
final topk = (r['attention_topk_items'] as List?)?.map((outer) => (outer as List).map((it) => it as Map<String,dynamic>).toList()).toList();
final colors = (r['attention_colors'] as List?)?.cast<String>();
final cmap = (r['attention_color_map'] as Map?)?.cast<String,dynamic>();
final rows = (r['attention_grid'] as List?)?[0] ?? 8; // fallback
final cols = (r['attention_grid'] as List?)?[1] ?? 8;

final tokenColors = parseTokenColorMap(cmap, tokens, colors);

// In build(): Stack(children: [ RawImage(image: bg), CustomPaint(painter: TokenMarkerPainter(backgroundImage: bg, topkItems: topk ?? [], tokens: tokens ?? [], tokenColors: tokenColors, rows: rows, cols: cols)) ])
*/

```

This file is a minimal integration example — you can expand it into a small Flutter widget that fetches the API, decodes, and animates markers per token.
