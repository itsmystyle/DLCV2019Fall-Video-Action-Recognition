# HW4 ― Videos
In this assignment, you will learn to perform both trimmed action recognition and temporal action segmentation in full-length videos.

<p align="center">
  <img width="750" height="250" src="https://lh3.googleusercontent.com/j48uA36UbZp3KR41opZUzntxhlJWoX_R5joeNsTGMN2_cSXI0UFNKuKVu8em_txzOIVbnU8p_oOb">
</p>

For more details, please refer to hw4 slides [this link](https://drive.google.com/file/d/1JZU1-MBGrclWOVfdNqAEn3dVZXt30DNa/view?usp=sharing).

# Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/dlcv-spring-2019/hw4-<username>.git
Note that you should replace `<username>` with your own GitHub username.

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `hw4_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/uc?export=download&id=1ncmqWLctmvecIXBdVng5cvbROoTWFSpE) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `hw4_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

For this dataset, the action labels are defined as below:

|       Action      | Label |
|:-----------------:|:-----:|
| Other             | 0     |
| Inspect/Read      | 1     |
| Open              | 2     |
| Take              | 3     |
| Cut               | 4     |
| Put               | 5     |
| Close             | 6     |
| Move Around       | 7     |
| Divide/Pull Apart | 8     |
| Pour              | 9     |
| Transfer          | 10    |

⚠️ Important Note⚠️
In this homework, you are not allowed to use any external datset except for the ImageNet dataset (You can load any model pre-trained only on the ImageNet dataset)

### Utility
We have also provided a Python script for reading video files and retrieving labeled videos as a dictionary. For more information, please read the comments in [`reader.py`](reader.py).

# Submission Rules
### Deadline
108/12/17 (Tue.) 03:00 AM

### Late Submission Policy
You have a five-day delay quota for the whole semester. Once you have exceeded your quota, the credit of any late submission will be deducted by 30% each day.

**NOTE:** To encourage students to submit homework on time, students using no more than three late-day quota will receive a bonus of two points in their final scores. Students using four late-day quota in this semester will receive a bonus of one point in their final scores.

Note that while it is possible to continue your work in this repository after the deadline, **we will by default grade your last commit before the deadline** specified above.

⚠️ Late Submission Forms⚠️ 
> **If your are going to submit your homework after the homework deadline, please follow the steps below carefully.**
> - You must fill in this form befofre the deadline of hw4 to notify the TAs that you have not finished your homework. Then, we will not grade your homewrok imediately. Please note that once you submit this form your homework will be classified as late submitted ones. Even if you send the TAs an email before the homework deadline, your homework will be regarded as one day late. Therefore, before filling in this form, you must be 100% sure that you really are going to submit your homework after the deadline.
>
>    https://forms.gle/zZYiqb7FfG51sQfK6
>
> - After you push your homework to the github repository, please fill in the form below immedaitely. We will calculate the number of late days according to the time we receive the response of this form. Please note that you will have access to this form after 108/12/17 10:00 A.M. Please also note that you can only fill in this form **once**, so you **must** make sure your homework is ready to be graded before submitting this form.
>
>    https://forms.gle/zaqey3xE2dNUriB99
>

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.

### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the root directory of this repository:
 1.   `hw4_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Grading*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 1.   `hw4_p1.sh`  
The shell script file for data preprocessing. This script takes as input two folders: the first one contains the video data, and the second one is where you should output the label file named `p1_valid.txt`.
 1.   `hw4_p2.sh`  
The shell script file for trimmed action recognition. This script takes as input two folders: the first one contains the video data, and the second one is where you should output the label file named `p2_result.txt`.
 1.   `hw4_p3.sh`  
The shell script file for temporal action segmentation. This script takes as input two folders: the first one contains the video data, and the second one is where you should output the label files named `<video_category>.txt`. Note that you should replace `<video_category>` accordingly, and a total of **7** files should be generated in this script.

We will run your code in the following manner:

**Problem 1**

    bash ./hw4_p1.sh $1 $2 $3
-   `$1` is the folder containing the ***trimmed*** validation videos (e.g. `TrimmedVideos/video/valid/`).
-   `$2` is the path to the ground truth label file for the videos (e.g. `TrimmedVideos/label/gt_valid.csv`).
-   `$3` is the folder to which you should output your predicted labels (e.g. `./output/`). Please do not create this directory in your shell script or python codes.

**Problem 2**

    bash ./hw4_p2.sh $1 $2 $3
-   `$1` is the folder containing the ***trimmed*** validation/test videos.
-   `$2` is the path to the ground truth label file for the videos (e.g. `TrimmedVideos/label/gt_valid.csv` or `TrimmedVideos/label/gt_test.csv`).
-   `$3` is the folder to which you should output your predicted labels (e.g. `./output/`). Please do not create this directory in your shell script or python codes.

**Problem 3**

    bash ./hw4_p3.sh $1 $2
-   `$1` is the folder containing the ***full-length*** validation videos.
-   `$2` is the folder to which you should output your predicted labels (e.g. `./output/`). Please do not create this directory in your shell script or python codes.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> - For the sake of conformity, please use the -**python3** command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.
> - You must **not** use commands such as **rm, sudo, CUDA_VISIBLE_DEVICES**, cp, mv, mkdir, cd, pip or other commands to change the Linux environment.
> - We will execute you code on Linux system, so please make sure you code can be executed on **Linux** system before submitting your homework.
> - **DO NOT** hard code any path in your file or script except for the path of your trained model.
> - The execution time of your testing code should not exceed an allowed maximum of **10 minutes**.
> - Use the wget command in your script to download you model files. Do not use the curl command.
> - **Do not delete** your trained model before the TAs disclose your homework score and before you make sure that your score is correct.
> - If you use matplotlib in your code, please add matplotlib.use(“Agg”) in you code or we will not be able to execute your code.
> - Do not use imshow() or show() in your code or your code will crash.
> - Use os.path.join to deal with path issues as often as possible.
> - Please do not upload your training information generated by tensorboard to github.

### Packages
This homework should be done using python3.6 and you can use all the python3.6 standard libraries. For a list of third-party packages allowed to be used in this assignment, please refer to the requirments.txt for more details.
You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

### Remarks
- If your model is larger than GitHub’s maximum capacity (100MB), you can upload your model to another cloud service (e.g. Dropbox). However, your shell script files should be able to download the model automatically. For a tutorial on how to do this using Dropbox, please click [this link](https://goo.gl/XvCaLR).

- **If we can not reproduce the scores or images in your report, you will get 0 points in the corresponding problem.**
- **If we can not execute your code, we will give you a chance to make minor modifications to your code. After you modify your code**
    - If we can execute your code and reproduce your results in the report, you will still receive a 30% penalty in your homework score.
    - If we can run your code but cannot reproduce your results in the report, you will get 0 points in the corresponding problem.
    - If we still cannot execute your code, you will get 0 in the corresponding problem.
# Q&A
If you have any problems related to HW4, you may
- Use TA hours (please check [course website](http://vllab.ee.ntu.edu.tw/dlcv.html) for time/location)
- Contact TAs by e-mail ([ntudlcvta2019@gmail.com](mailto:ntudlcvta2019@gmail.com))
- Post your question in the FB club.

