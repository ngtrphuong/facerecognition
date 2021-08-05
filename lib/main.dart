// Google ML Vision Face Detection and recognition app
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// @dart=2.9

import 'package:flutter/material.dart';

import 'camera_detector.dart';

void main() {
  runApp(
    MaterialApp(
      routes: <String, WidgetBuilder>{
        '/': (BuildContext context) => const CameraDetector(),
      },
      debugShowCheckedModeBanner: false,
    ),
  );
}

class _FaceRecognition extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => _FaceRecognitionState();
}

class _FaceRecognitionState extends State<_FaceRecognition> with WidgetsBindingObserver {
  static final List<String> _faceRecogWidgetName = <String>[
    '$CameraDetector',
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Recognition App'),
      ),
      body: ListView.builder(
        itemCount: _faceRecogWidgetName.length,
        itemBuilder: (BuildContext context, int index) {
          final String widgetName = _faceRecogWidgetName[index];

          return Container(
            decoration: const BoxDecoration(
              border: Border(bottom: BorderSide(color: Colors.grey)),
            ),
            child: ListTile(
              title: Text(widgetName),
              onTap: () => Navigator.pushNamed(context, '/$widgetName'),
            ),
          );
        },
      ),
    );
  }
}
