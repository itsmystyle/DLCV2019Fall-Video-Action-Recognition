import argparse

from modules.reader import getVideoList

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation of HW4.")
    parser.add_argument("problem", type=str, help="Problem 1, 2 or 3.")
    parser.add_argument("gt", type=str, help="Groundthruth file.")
    parser.add_argument("pred", type=str, help="Predicted file.")

    args = parser.parse_args()

    if args.problem != "3":
        gt = getVideoList(args.gt)["Action_labels"]
        with open(args.pred, "r") as fin:
            pred = fin.readlines()

        assert len(gt) == len(pred), "Number of ground-truth and predicts not same!"

        acc = 0
        for _g, _p in zip(gt, pred):
            if _g == _p.strip():
                acc += 1
        print(acc / len(gt))

    else:
        pass
