# TODO: create shell script for Problem 2
wget 'https://www.dropbox.com/s/ndmet4hvk3h1hhl/model_best_RCNN.pth.tar?dl=1' -O models/RCNN/model_best_RCNN.pth.tar
python3 predict_p1p2.py RCNN $1 $2 models/RCNN/model_best_RCNN.pth.tar $3 --sorting --batch_size 1