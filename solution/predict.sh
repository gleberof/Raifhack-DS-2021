#!/bin/bash
python predict_lgbm.py
python predict_tabnet.py
python blend.py