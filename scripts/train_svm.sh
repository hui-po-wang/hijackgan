ATTRIBUTE_NAME="pale_skin"
rm -r ../boundaries/stylegan_celebahq_"$ATTRIBUTE_NAME"
python ../train_boundary.py \
    -o ../boundaries/stylegan_celebahq_"$ATTRIBUTE_NAME" \
    -c ../data/svm_train/13_z.npy \
    -s ../data/svm_train/13_labels.npy
