// Google ML Vision Face Detection and recognition app
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// @dart=2.9

import 'dart:ui' as ui;
import 'dart:math';
import 'dart:io';
import 'package:google_ml_vision/google_ml_vision.dart';
import 'package:flutter/material.dart';

enum Detector {
  face,
}

const List<Point<int>> faceMaskConnections = [
  Point(0, 4),
  Point(0, 55),
  Point(4, 7),
  Point(4, 55),
  Point(4, 51),
  Point(7, 11),
  Point(7, 51),
  Point(7, 130),
  Point(51, 55),
  Point(51, 80),
  Point(55, 72),
  Point(72, 76),
  Point(76, 80),
  Point(80, 84),
  Point(84, 72),
  Point(72, 127),
  Point(72, 130),
  Point(130, 127),
  Point(117, 130),
  Point(11, 117),
  Point(11, 15),
  Point(15, 18),
  Point(18, 21),
  Point(21, 121),
  Point(15, 121),
  Point(21, 25),
  Point(25, 125),
  Point(125, 128),
  Point(128, 127),
  Point(128, 29),
  Point(25, 29),
  Point(29, 32),
  Point(32, 0),
  Point(0, 45),
  Point(32, 41),
  Point(41, 29),
  Point(41, 45),
  Point(45, 64),
  Point(45, 32),
  Point(64, 68),
  Point(68, 56),
  Point(56, 60),
  Point(60, 64),
  Point(56, 41),
  Point(64, 128),
  Point(64, 127),
  Point(125, 93),
  Point(93, 117),
  Point(117, 121),
  Point(121, 125),
];

class FaceDetectorLandmarkPainter extends CustomPainter {
  FaceDetectorLandmarkPainter(this.absoluteImageSize, this.results);

  final Size absoluteImageSize;
  //final List<Face> faces;

  dynamic results;
  //Face face;

  @override
  void paint(Canvas canvas, Size size) {
    final double scaleX = size.width / absoluteImageSize.width;
    final double scaleY = size.height / absoluteImageSize.height;

    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.red;

    final Paint linePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.green;
    for (String label in results.keys) {
      for (final Face face in results[label]) {
        final contour = face.getContour((FaceContourType.allPoints));
        if (Platform.isAndroid) {
          canvas.drawPoints(
              ui.PointMode.points,
              contour.positionsList
                  .map((offset) =>
                  Offset(size.width - (offset.dx * scaleX), offset.dy * scaleY))
                  .toList(),
              paint);
        } else if (Platform.isIOS) {
          canvas.drawPoints(
              ui.PointMode.points,
              contour.positionsList
                  .map((offset) =>
                  Offset((offset.dx * scaleX), offset.dy * scaleY))
                  .toList(),
              paint);
        }
        for (final connection in faceMaskConnections) {
          if (Platform.isAndroid) {
            if(contour.positionsList != null && contour.positionsList.length != 0) {
            double tmpDxConnectionX = size.width -
                contour.positionsList[connection.x]
                    .scale(scaleX, scaleY)
                    .dx;
            double tmpDyConnectionX = contour.positionsList[connection.x]
                .scale(scaleX, scaleY)
                .dy;
            Offset a = Offset(tmpDxConnectionX, tmpDyConnectionX);
            double tmpDxConnectionY = size.width -
                contour.positionsList[connection.y]
                    .scale(scaleX, scaleY)
                    .dx;
            double tmpDyConnectionY = contour.positionsList[connection.y]
                .scale(scaleX, scaleY)
                .dy;
            Offset b = Offset(tmpDxConnectionY, tmpDyConnectionY);
            canvas.drawLine(a, b, paint);
          }
          } else if (Platform.isIOS) {
            canvas.drawLine(
                contour.positionsList[connection.x].scale(scaleX, scaleY),
                contour.positionsList[connection.y].scale(scaleX, scaleY),
                paint);
          }
        }
        if (label == "NOT RECOGNIZED") {
          linePaint.color = Colors.purple;
        }
        if (Platform.isAndroid) {
          canvas.drawRRect(
              _scaleRect(
                  rect: face.boundingBox,
                  imageSize: absoluteImageSize,
                  widgetSize: size,
                  scaleX: scaleX,
                  scaleY: scaleY),
              linePaint);
        } else if (Platform.isIOS) {
          canvas.drawRect(
              _scaleRect(
                  rect: face.boundingBox,
                  imageSize: absoluteImageSize,
                  widgetSize: size,
                  scaleX: scaleX,
                  scaleY: scaleY),
              linePaint);
        }
        TextSpan span = new TextSpan(
            style: new TextStyle(color: Colors.deepOrange[600], fontSize: 20,
                fontWeight: FontWeight.bold),
            text: label
        );
        TextPainter textPainter = new TextPainter(
            text: span,
            textAlign: TextAlign.left,
            textDirection: TextDirection.ltr);
        textPainter.layout();
        if (Platform.isAndroid) {
          textPainter.paint(
              canvas,
              new Offset(
                  size.width - (face.boundingBox.left.toDouble()) * scaleX,
                  (face.boundingBox.top.toDouble() ) * scaleY));
        } else if (Platform.isIOS) {
          textPainter.paint(
              canvas,
              new Offset(
                  (face.boundingBox.left.toDouble()) * scaleX,
                  (face.boundingBox.top.toDouble() ) * scaleY));
        }
      }
    }
  }

  @override
  bool shouldRepaint(FaceDetectorLandmarkPainter oldDelegate) {
    return oldDelegate.absoluteImageSize != absoluteImageSize ||
        oldDelegate.results != results;
  }
}
//Draw Face rectangle with Name on it
class FaceDetectorNormalPainter extends CustomPainter {
  FaceDetectorNormalPainter(this.imageSize, this.results);
  final Size imageSize;
  double scaleX, scaleY;
  dynamic results;
  Face face;
  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.greenAccent;
    for (String label in results.keys) {
      for (Face face in results[label]) {
        // face = results[label];
        scaleX = size.width / imageSize.width;
        scaleY = size.height / imageSize.height;
        if (label == "NOT RECOGNIZED") {
          paint.color = Colors.purple;
        }
        if (Platform.isAndroid) {
          canvas.drawRRect(
              _scaleRect(
                  rect: face.boundingBox,
                  imageSize: imageSize,
                  widgetSize: size,
                  scaleX: scaleX,
                  scaleY: scaleY),
              paint);
        } else if (Platform.isIOS) {
          canvas.drawRect(
              _scaleRect(
                  rect: face.boundingBox,
                  imageSize: imageSize,
                  widgetSize: size,
                  scaleX: scaleX,
                  scaleY: scaleY),
              paint);
        }
        TextSpan span = new TextSpan(
            style: new TextStyle(color: Colors.red[600], fontSize: 20,
                fontWeight: FontWeight.bold),
            text: label);
        TextPainter textPainter = new TextPainter(
            text: span,
            textAlign: TextAlign.left,
            textDirection: TextDirection.ltr);
        textPainter.layout();
        if(Platform.isIOS) {
        textPainter.paint(
            canvas,
            new Offset(
                (10 + face.boundingBox.left.toDouble()) * scaleX,
                (face.boundingBox.top.toDouble() - 15) * scaleY));
        } else if (Platform.isAndroid) {
          textPainter.paint(
              canvas,
              new Offset(
                  size.width - (70 + face.boundingBox.left.toDouble()) * scaleX,
                  (face.boundingBox.top.toDouble() - 0) * scaleY));
        }
      }
    }
  }

  @override
  bool shouldRepaint(FaceDetectorNormalPainter oldDelegate) {
    return oldDelegate.imageSize != imageSize || oldDelegate.results != results;
  }
}

dynamic _scaleRect(
    {@required Rect rect,
      @required Size imageSize,
      @required Size widgetSize,
      double scaleX,
      double scaleY}) {
  RRect _rRect;
  Rect _rect;
  dynamic result;
  if (Platform.isAndroid) {
    _rRect = RRect.fromLTRBR(
        (widgetSize.width - rect.left.toDouble() * scaleX),
        rect.top.toDouble() * scaleY,
        widgetSize.width - rect.right.toDouble() * scaleX,
        rect.bottom.toDouble() * scaleY,
        Radius.circular(10));
    result = _rRect;
  } else if (Platform.isIOS) {
    _rect = Rect.fromLTRB(
      rect.left * scaleX,
      rect.top * scaleY,
      rect.right * scaleX,
      rect.bottom * scaleY,
    );
    result = _rect;
  }
  return result;
}
