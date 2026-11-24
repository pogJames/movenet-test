import argparse
import os
import time

import cv2
import numpy as np
import ai_edge_litert.interpreter as tflite

DELEGATE = "/usr/local/lib/litert_delegate.so"
LINE_COLOR = (255, 128, 0)
POINT_COLOR = (0, 0, 255)

parser = argparse.ArgumentParser(description="MoveNet pose estimation test")
parser.add_argument("--model", default="movenet.tflite", help="path to .tflite model")
parser.add_argument("--image", default="example_input.jpg", help="path to input image")
parser.add_argument("--runs", type=int, default=50, help="timed inference runs")
parser.add_argument("--warmup", type=int, default=5, help="untimed runs first")
parser.add_argument("--cpu", action="store_true", help="run on CPU instead of the NPU")
args = parser.parse_args()

MODEL_FILENAME = args.model
IMAGE_FILENAME = args.image

# output name from input name: photos/worker1.jpg -> photos/worker1_output.jpg
base, ext = os.path.splitext(IMAGE_FILENAME)
OUTPUT_FILENAME = f"{base}_output{ext}"

# keypoints definition
# https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#keypoint-diagram
keypoints_def = [
    {'label': 'nose',               'connections': [1, 2,]},
    {'label': 'left_eye',           'connections': [0, 3,]},
    {'label': 'right_eye',          'connections': [0, 4,]},
    {'label': 'left_ear',           'connections': [1,]},
    {'label': 'right_ear',          'connections': [2,]},
    {'label': 'left_shoulder',      'connections': [6, 7, 11,]},
    {'label': 'right_shoulder',     'connections': [5, 8, 12,]},
    {'label': 'left_elbow',         'connections': [5, 9,]},
    {'label': 'right_elbow',        'connections': [6, 10,]},
    {'label': 'left_wrist',         'connections': [7,]},
    {'label': 'right_wrist',        'connections': [8,]},
    {'label': 'left_hip',           'connections': [5, 12, 13,]},
    {'label': 'right_hip',          'connections': [6, 11, 14,]},
    {'label': 'left_knee',          'connections': [11, 15,]},
    {'label': 'right_knee',         'connections': [12, 16,]},
    {'label': 'left_ankle',         'connections': [13,]},
    {'label': 'right_ankle',        'connections': [14,]},
    ]

connections = [(i, j) for i in range(len(keypoints_def))
               for j in keypoints_def[i]["connections"]]

if args.cpu:
    print("Backend: CPU")
    interpreter = tflite.Interpreter(MODEL_FILENAME)
else:
    print(f"Backend: NPU ({DELEGATE})")
    interpreter = tflite.Interpreter(
        MODEL_FILENAME,
        experimental_delegates=[tflite.load_delegate(DELEGATE)])
interpreter.allocate_tensors()


def load_image(filename):
    orig_image = cv2.imread(filename, 1)
    if orig_image is None:
        raise SystemExit(f"Could not read image: {filename}")
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (192, 192))
    image = image[..., ::-1]   # as in NXP's original; note this flips RGB back to BGR
    image = np.expand_dims(image, axis=0)
    return orig_image, image


def run_inference(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = image.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index']).astype(np.float32)

    return out


orig_image, processed_image = load_image(IMAGE_FILENAME)

# warm-up: first runs include one-off setup costs
for _ in range(args.warmup):
    run_inference(interpreter, processed_image)

times = []
for _ in range(args.runs):
    start = time.perf_counter()
    out = run_inference(interpreter, processed_image)[0, 0, ...]
    times.append((time.perf_counter() - start) * 1000)

times = np.array(times)
print(f"Inference over {args.runs} runs (after {args.warmup} warm-up):")
print(f"  mean   {times.mean():6.2f} ms  ({1000 / times.mean():.0f} inferences/s)")
print(f"  median {np.median(times):6.2f} ms")
print(f"  min    {times.min():6.2f} ms")
print(f"  p90    {np.percentile(times, 90):6.2f} ms")
print(f"  stdev  {times.std():6.2f} ms")

# keypoints are [y, x, confidence], normalised 0-1; scale to image pixels
h, w = orig_image.shape[0:2]
out[:, 0] *= h
out[:, 1] *= w

for c in connections:
    i = c[0]
    j = c[1]
    cv2.line(orig_image,
             (int(out[i, 1]), int(out[i, 0])),
             (int(out[j, 1]), int(out[j, 0])), LINE_COLOR, 5)

for i in range(out.shape[0]):
    cv2.circle(orig_image,
               (int(out[i, 1]), int(out[i, 0])), 5, POINT_COLOR, 10)

cv2.imwrite(OUTPUT_FILENAME, orig_image)
print("Saved", OUTPUT_FILENAME)