# TODO: create shell script for Problem 1
wget 'https://www.dropbox.com/s/mme31p01aa1hi8k/model_best_SCNN.pth.tar?dl=1' -O models/SCNN/model_best_SCNN.pth.tar
python3 predict_p1p2.py SCNN $1 $2 models/SCNN/model_best_SCNN.pth.tar $3 --sorting --batch_size 4