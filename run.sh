#!/bin/bash

# Run Random, Basic and VggFace
cd src
python main.py

# Run Facenet
cd ../facenet
python src/classifier.py CLASSIFY ../faces_aligned/valid ../saved_models/20170512-110547/ ../saved_models/my_facenet.pkl