. ./scripts/CONFIG

python train.py \
    --img_root ${FLOWERS_IMG_ROOT} \
    --caption_root ${FLOWERS_CAPTION_ROOT} \
    --trainclasses_file trainvalclasses.txt \
    --fasttext_model ${FASTTEXT_MODEL} \
    --text_embedding_model ./models/text_embedding_flower.pth \
    --save_filename ./models/flowers_vgg.pth \
    --use_vgg