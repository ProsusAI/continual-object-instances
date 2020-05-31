#!/bin/bash
# A script to download the Cars3D dataset
# Copyright (c) 2016 Taehoon Kim
# From https://github.com/carpedm20/visual-analogy-tensorflow

wget http://www-personal.umich.edu/~reedscot/files/nips2015-analogy-data.tar.gz
tar -zxvf nips2015-analogy-data.tar.gz
rm nips2015-analogy-data.tar.gz