import os
import argparse

from modules.reader import getVideoList

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation of HW4.")
    parser.add_argument("problem", type=str, help="Problem 1, 2 or 3.")
    parser.add_argument("gt", type=str, help="Ground-truth file.")
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
        categories = sorted(os.listdir(args.gt))

        acc = {}

        for category in categories:
            with open(os.path.join(args.gt, category), "r") as fin:
                targets = fin.readlines()
            with open(os.path.join(args.pred, category), "r") as fin:
                preds = fin.readlines()

            if category not in acc:
                acc[category] = {"n": 0, "n_correct": 0}

            assert len(targets) == len(
                preds
            ), "Number of ground-truth and predicts not same!"

            acc[category]["n"] = len(targets)

            for _g, _p in zip(targets, preds):
                if _g.strip() == _p.strip():
                    acc[category]["n_correct"] += 1

        n_total = sum([v["n"] for k, v in acc.items()])
        n_correct = sum([v["n_correct"] for k, v in acc.items()])

        print(n_correct / n_total)
