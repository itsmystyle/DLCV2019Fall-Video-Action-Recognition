# TODO: create shell script for Problem 3
wget 'https://www.dropbox.com/s/m69g5us5oqi3kfu/model_best_LRCNN.pth.tar?dl=1' -O models/Full_LRCNN/model_best_LRCNN.pth.tar
python3 predict_p3.py $1 models/Full_LRCNN/model_best_LRCNN.pth.tar $2 --sorting --batch_size 1