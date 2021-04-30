#!/bin/bash

python test.py \
  --model weights/REGDet_pyramidbox.pth \
  --pred_data output_data/REGDet_pyramidbox.json 
