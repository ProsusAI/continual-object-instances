#!/bin/bash
# A script to download the Cars3D dataset
# Copyright (c) 2016 Taehoon Kim
# From https://github.com/carpedm20/visual-analogy-tensorflow

#wget http://www-personal.umich.edu/~reedscot/files/nips2015-analogy-data.tar.gz
# faster link
wget http://www.scottreed.info/files/nips2015-analogy-data.tar.gz
tar -zxvf nips2015-analogy-data.tar.gz
rm nips2015-analogy-data.tar.gz

# render the 3d dataset using matlab or octave
mv data/cars .
rm -fr data
octave Cars3D/render.m
mv images Cars3D/images
