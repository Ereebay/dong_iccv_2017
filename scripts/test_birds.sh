#!/usr/bin/env bash
. ./scripts/CONFIG

python test.py \
    --img_root ./test/birds \
    --text_file ./test/text_birds.txt \
    --fasttext_model ${FASTTEXT_MODEL} \
    --text_embedding_model ./models/text_embedding_birds.pth \
    --generator_model ./models/birds_vgg.pth \
    --output_root ./test/result_birds \
    --use_vgg