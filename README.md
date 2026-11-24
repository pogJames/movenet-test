# MoveNet pose estimation test (i.MX 93 / Matrix-800)

Runs Google's **MoveNet SinglePose Lightning** (int8) on one image and draws the detected body pose on it. It's a quick check that the model, LiteRT and the Ethos-U NPU delegate all work on the board.

The model finds **one person** and returns 17 body keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles), each with a confidence score. The script draws them as red dots joined into an orange skeleton.

Based on NXP's [eIQ Model Zoo MoveNet example](https://github.com/NXP/eiq-model-zoo/tree/main/tasks/vision/pose-estimation/movenet).

## Requirements

- `movenet.tflite` in this folder (downloaded by `recipe.sh`)
- Virtual environment (.venv) with `pip install -r requirements.txt`
- Ethos-U delegate at `/usr/local/lib/litert_ethosu_delegate.so`

## Usage

```bash
source .venv/bin/activate

python example.py                                   # movenet.tflite + example_input.jpg
python example.py --image worker1.jpg               # your own image file
python example.py --model model_imx93/movenet_vela.tflite --image photos/fall.jpg
python example.py --help
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | `movenet.tflite` | `.tflite` model to run (plain or Vela-compiled) |
| `--image` | `example_input.jpg` | input image |
| `--runs` | `50` | timed inference runs |
| `--warmup` | `5` | untimed runs before timing |
| `--cpu` | off | run on the CPU instead of the NPU |

It prints the mean, median, min, p90 and standard deviation of the inference time. The mean is comparable to `benchmark_model`'s `avg`, but includes a little Python overhead per run.

The result is saved next to the input with `_output` added to the name: `photos/fall.jpg` becomes `photos/fall_output.jpg`.

## NPU or CPU

The script runs on the NPU by default. Add `--cpu` to run on the CPU for comparison. The delegate path is set by `DELEGATE` at the top of `example.py`.

Expected on the Matrix-800:

| Backend | Mean inference | Log line |
|---|---|---|
| NPU | ~7–9 ms | `EthosuDelegate: 116 nodes delegated out of 157 nodes with 7 partitions` |
| CPU | ~42–45 ms | `Created TensorFlow Lite XNNPACK delegate for CPU` only |

## Notes

- **Single person only.** With several people in the image, the skeleton may belong to one of them or mix points from several.
- Every keypoint is drawn, even low-confidence ones, so points the model didn't really see can appear in odd places.
- There's no display output. View the saved image on your PC: `scp root@matrix800:<path>/fall_output.jpg .`