# Session 3

Train a model remotely using Google Cloud Compute Engine.

The code in this session **already works**: you don't need to write any Python. The goal of today is to learn how to connect to a remote machine (a Google Cloud VM) over SSH, move your code and data there, and train the model on it.

## Dataset

We will use the cars vs. flowers dataset. You can download it [here](https://www.kaggle.com/olavomendes/cars-vs-flowers) (direct [download](https://www.kaggle.com/api/v1/datasets/download/olavomendes/cars-vs-flowers)).

Unzip it in the root of this repository. You should end up with this structure:

```
dataset/cars_vs_flowers/training_set/car
dataset/cars_vs_flowers/training_set/flower
dataset/cars_vs_flowers/test_set/car
dataset/cars_vs_flowers/test_set/flower
```

If you put the dataset somewhere else, change `dataset_path` in the `config` at the bottom of `main.py`.

## Installation
### With Conda
Create a conda environment by running

```bash
conda create --name mlops-session3 python=3.10
```

Then, activate the environment

```bash
conda activate mlops-session3
```

and install the dependencies

```bash
pip install -r session-3/requirements.txt
```

## Running the project

To run the project, run this from the root of the repository
```bash
python session-3/main.py
```

You should see the loss and accuracy of every epoch, and at the end the trained model is saved as `model.pth`.

## Tasks

Replace `[USERNAME]` and `[EXTERNAL_IP]` with your own values. Follow the slides for the details of each step.


### 1. Create an SSH key pair

On your computer:

```bash
ssh-keygen -t rsa -f ~/.ssh/ics2026 -C [USERNAME]
```

This creates two files: `~/.ssh/ics2026` (the **private** key, never share it) and `~/.ssh/ics2026.pub` (the **public** key).

### 2. Add the public key to the VM

Print your public key and copy it:

```bash
cat ~/.ssh/ics2026.pub
```

In the Google Cloud console go to *Compute Engine* > *VM instances*, click on your instance, click *Edit*, scroll down to *SSH keys*, paste the key and save.

### 3. Connect to the VM

```bash
ssh -i ~/.ssh/ics2026 [USERNAME]@[EXTERNAL_IP]
```

You are now in a terminal on the remote machine. Type `exit` to go back to your computer.

### 4. Install Python on the VM

Connected to the VM, install miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

Close the connection (`exit`) and connect again so `conda` is available.

### 5. Move the code and the dataset to the VM

From **your computer**, in the root of this repository:

```bash
ssh -i ~/.ssh/ics2026 [USERNAME]@[EXTERNAL_IP] "mkdir -p ~/mlops"
scp -i ~/.ssh/ics2026 -r session-3 dataset [USERNAME]@[EXTERNAL_IP]:~/mlops
```

The first command creates the `~/mlops` folder on the VM, and the second one copies the code and the dataset into it.

On Linux, Mac or WSL you can use `rsync` instead of `scp`, which is faster.

```bash
rsync -r -e "ssh -i ~/.ssh/ics2026" session-3 dataset [USERNAME]@[EXTERNAL_IP]:~/mlops
```

### 6. Train on the VM

Connect to the VM again, and follow the same [Installation](#installation) and [Running the project](#running-the-project) steps as on your computer, from the `~/mlops` folder:

```bash
cd ~/mlops
conda create --name mlops-session3 python=3.10
conda activate mlops-session3
pip install -r session-3/requirements.txt
python session-3/main.py
```

### 7. Use VSCode on the VM

Install the *Remote - SSH* extension in VSCode. From the command palette (F1), run *Remote-SSH: Add New SSH Host* and enter the same `ssh` command you used in step 3. Then run *Remote-SSH: Connect to Host* and open the `~/mlops` folder.

Now you can edit, run and debug the files on the VM as if they were on your computer. Install the Python extension on the server and select the `mlops-session3` interpreter, the same as we did in session 1.

## Optional task

If you have finished, you can try to improve the accuracy: add data augmentation to the transforms in `main.py` (for example `transforms.RandomHorizontalFlip()`), or change the layers in `model.py`.

## Possible issues

If you observe the following error in the (cloud) computer:

```
Error: OSError: [Errno 28] No space left on device
```

even though the VM disk had plenty of space, the installation failed because pip downloads packages to the /tmp folder, which on many VM's has a small size limit. Large packages (like PyTorch) can exceed that limit.

**Quick solution** install the packages using conda instead, which internally handles temporary files differently and avoids the issue:

```
conda install numpy matplotlib Pillow pytorch torchvision cpuonly -c pytorch
```

If `ssh` answers `Permission denied (publickey)`, check that the `[USERNAME]` you connect with is the same one that appears at the end of your public key, and that you are passing the right key with `-i`.

On Windows, if `ssh` is not recognized as a command in PowerShell, use *Git Bash* instead, which includes it.
