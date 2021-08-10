// @dart=2.9
import 'dart:core';
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as imglib;
//import 'package:quiver/collection.dart';

class FaceAntiSpoofing {
  FaceAntiSpoofing._();

  static const String MODEL_FILE = "FaceAntiSpoofing.tflite";
  static const INPUT_IMAGE_SIZE = 256; // The width and height of the placeholder image that needs feed data
  static const THRESHOLD = 0.2; // Set a threshold value, greater than this value is considered an attack
  static const ROUTE_INDEX = 6; // Route index observed during training
  static const LAPLACE_THRESHOLD = 50; // Laplace sampling threshold
  static const LAPLACIAN_THRESHOLD = 1000; // Picture clarity judgment threshold
  static tfl.Interpreter interpreter;

  /*
   * Live detection
   */
  static String antiSpoofing(imglib.Image bitmapCrop1, imglib.Image bitmapCrop2) {
    if (bitmapCrop1 == null || bitmapCrop2 == null) {
      print("Please detect the face first");
      return "Nothing";
    }

    // Judge the clarity of the picture before live detection
    var laplace1 = laplacian(bitmapCrop1);

    String text = "Sharpness detection result left：" + laplace1.toString();
    if (laplace1 < LAPLACIAN_THRESHOLD) {
      text = text + "，" + "False";
    } else {
      var start = DateTime.now().microsecondsSinceEpoch;

      // Live detection
      var score1 = _antiSpoofing(bitmapCrop1);

      var end = DateTime.now().microsecondsSinceEpoch;

      text = "Sharpness detection result left：" + score1.toString();
      if (score1 < THRESHOLD) {
        text = text + "，" + "True";
      } else {
        text = text + "，" + "False";
      }
      text = text + ".Time consuming: " + (end - start).toString();
    }
    print(text);

    // Judge the clarity of the second picture before live detection
    var laplace2 = laplacian(bitmapCrop2);

    String text2 = "Sharpness detection result left：" + laplace2.toString();
    if (laplace2 < LAPLACIAN_THRESHOLD) {
      text2 = text2 + "，" + "False";
    } else {
      // Live detection
      var score2 = _antiSpoofing(bitmapCrop2);
      text2 = "Liveness test result right：" + score2.toString();
      if (score2 < THRESHOLD) {
        text2 = text2 + "，" + "True";
      } else {
        text2 = text2 + "，" + "False";
      }
    }
    print(text2);
    return text2;
  }

  static Future loadSpoofModel() async {
    try {
      interpreter = await tfl.Interpreter.fromAsset(MODEL_FILE);

      print('**********\n Loaded successfully model $MODEL_FILE \n*********\n');
    } catch (e) {
      print('Failed to load model.');
      print(e);
    }
  }

  /*
   * Live detection
   * @param bitmap
   * @return score
   */
  static double _antiSpoofing(imglib.Image bitmap) {
    // Resize the face to a size of 256X256, because the shape of the placeholder that needs feed data below is (1, 256, 256, 3)
    imglib.Image bitmapScale = imglib.copyResizeCropSquare(bitmap, INPUT_IMAGE_SIZE);

    var img = normalizeImage(bitmapScale);

    List input = new List.generate(1, (index) => List.filled(8, 0.0),growable: true);

    input[0] = img.reshape([1,INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE,3]);

    List clssPred = new List.generate(1, (index) => List.filled(8, 0.0));
    List leafNodeMask = new List.generate(1, (index) => List.filled(8, 0.0));

    Map outputs = new Map<int, Object>();

    outputs[interpreter.getOutputIndex("Identity")] = clssPred;
    outputs[interpreter.getOutputIndex("Identity_1")] = leafNodeMask;

    if (input.isNotEmpty && outputs.isNotEmpty && input.length > 0 && outputs.length > 0) {
      interpreter.runForMultipleInputs(input, outputs);

      print("FaceAntiSpoofing" + "[" + clssPred[0][0].toString() + ", " +
          clssPred[0][1].toString() + ", "
          + clssPred[0][2].toString() + ", " + clssPred[0][3].toString() +
          ", " + clssPred[0][4].toString() + ", "
          + clssPred[0][5].toString() + ", " + clssPred[0][6].toString() +
          ", " + clssPred[0][7].toString() + "]\n");
      print("FaceAntiSpoofing" + "[" + leafNodeMask[0][0].toString() + ", " +
          leafNodeMask[0][1].toString() + ", "
          + leafNodeMask[0][2].toString() + ", " +
          leafNodeMask[0][3].toString() + ", " + leafNodeMask[0][4].toString() +
          ", "
          + leafNodeMask[0][5].toString() + ", " +
          leafNodeMask[0][6].toString() + ", " + leafNodeMask[0][7].toString() +
          "]\n");

      return leafScore1(clssPred, leafNodeMask);
    } else {
      print("ERROR in input/output values");
      return -1;
    }
  }

  static dynamic leafScore1(var clssPred, var leafNodeMask) {
    var score = 0.0;
    for (var i = 0; i < 8; i++) {
      var absVar = (clssPred[0][i]).abs();
      score += absVar * leafNodeMask[0][i];
    }
    return score;
  }

  static dynamic leafScore2(var clssPred) {
    return clssPred[0][ROUTE_INDEX];
  }

  /*
   * Normalize the picture to [0, 1]
   * @param bitmap
   * @return
   */
  static Float32List normalizeImage(imglib.Image bitmap) {
    var h = bitmap.height;
    var w = bitmap.width;
    var convertedBytes = Float32List(1 * h * w * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    var imageStd = 128;
    var pixelIndex = 0;

    for (var i = 0; i < h; i++) { // Note that it is height first and then width
      for (var j = 0; j < w; j++) {
        var pixel = bitmap.getPixel(j, i);
        buffer[pixelIndex++] =  (imglib.getRed(pixel) - imageStd) / imageStd;
        buffer[pixelIndex++] =  (imglib.getGreen(pixel) - imageStd) / imageStd;
        buffer[pixelIndex++] =  (imglib.getBlue(pixel) - imageStd) / imageStd;

      }
    }
    return convertedBytes.buffer.asFloat32List();
  }

  /*
   * Laplacian algorithm to calculate clarity
   * @param bitmap
   * @return Fraction
   */
  static dynamic laplacian(imglib.Image bitmap) {
    // Resize the face to a size of 256X256, because the shape of the placeholder that needs feed data below is (1, 256, 256, 3)
    imglib.Image bitmapScale = imglib.copyResizeCropSquare(bitmap, INPUT_IMAGE_SIZE);

    var laplace = [[0, 1, 0], [1, -4, 1], [0, 1, 0]];
    int size = laplace.length;
    var img = imglib.grayscale(bitmapScale);
    int height = img.height;
    int width = img.width;

    int score = 0;
    for (int x = 0; x < height - size + 1; x++){
      for (int y = 0; y < width - size + 1; y++){
        int result = 0;
        // Convolution operation on size*size area
        for (int i = 0; i < size; i++){
          for (int j = 0; j < size; j++){
            result += (img.getPixel(x + i,y + j) & 0xFF) * laplace[i][j];
          }
        }
        if (result > LAPLACE_THRESHOLD) {
          score++;
        }
      }
    }
    return score;
  }
}