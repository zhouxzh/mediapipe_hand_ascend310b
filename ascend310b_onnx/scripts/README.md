# Scripts

```bash
# PC check
python scripts/run_image_onnx.py data/smoke_images/images/train_palm_ac44e9bd-97a1-4f28-8398-f825842fc59d.jpg

# Convert on a CANN machine or the 310B board
bash scripts/convert_onnx_to_om.sh full Ascend310B1

# Board inference
python3 scripts/run_image_ascend.py data/smoke_images/images/train_palm_ac44e9bd-97a1-4f28-8398-f825842fc59d.jpg
python3 scripts/run_video_ascend_tracking.py /path/to/video.mp4 --max-frames 100
```

