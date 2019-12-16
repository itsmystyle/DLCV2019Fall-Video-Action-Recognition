import matplotlib.pyplot as plt


with open("../hw4_data/FullLengthVideos/labels/valid/OP01-R07-Pizza.txt", "r") as fin:
    gt = fin.readlines()

with open("../output/OP01-R07-Pizza.txt", "r") as fin:
    pd = fin.readlines()

n_frames = 1320
plt.figure(figsize=(18, 4))
for i, (g, p) in enumerate(zip(gt[50:n_frames], pd[50:n_frames])):
    widths = 1
    starts = widths * i
    plt.barh(
        "Prediction",
        widths,
        left=starts,
        height=0.45,
        color="C{}".format(p.strip()) if p.strip() != "10" else "black",
    )
    plt.barh(
        "Ground-Truth",
        widths,
        left=starts,
        height=0.45,
        color="C{}".format(g.strip()) if p.strip() != "10" else "black",
    )
plt.xlim(50, 1320)
plt.savefig("../p3_visualize.png")
plt.close()
