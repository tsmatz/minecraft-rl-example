# Reinforcement Learning in Minecraft (Project Malmo Example)

This example code trains an agent in Minecraft with reinforcement learning. (Here I have used RLlib.)<br>
In this example, a maze (in which, the path is randomized) is given in each episode, and the agent will learn to reach to a goal block using the observed frame pixels (84 x 84 x 3 channels).

See [here](https://tsmatz.wordpress.com/2020/07/09/minerl-and-malmo-reinforcement-learning-in-minecraft/) for the tutorial of Project Malmo, which is built on a modded Minecraft by Microsoft Research.

This Readme provides instructions for running this example.

## 1. Setup prerequisite software in Ubuntu ##

In this example, I assume Ubuntu 20.04 with real monitor (which is used to show Minecraft UI) to run the training. (I have used **Ubuntu Server 20.04 LTS** in Microsoft Azure.)<br>
You can also join into the same game with your own Minecraft PC client, if needed. (See [here](https://tsmatz.wordpress.com/2020/07/09/minerl-and-malmo-reinforcement-learning-in-minecraft/) for details.)

Now let's start to set up your Ubuntu environment.

<blockquote>
Note : I have run and trained the agent in GPU utilized VM (instance).<br>
When you run on NVIDIA GPU-utilized machine to speed up, please setup GPU drivers and libraries as follows.

I note that you should install correct version of drivers and libraries. (In this settings, we'll use CUDA version 11.0 and cuDNN versioin 8.0, since we will use TensorFlow 2.4.1. See https://www.tensorflow.org/install/source#gpu for details about compatible drivers in TensorFlow.)

For preparation, install ```gcc``` and ```make``` tools.

```
sudo apt-get update
sudo apt-get install build-essential
# # or install individual packages as follows
# sudo apt-get install -y gcc
# sudo apt-get install -y make
```

Install CUDA 11.0 by running the following command. (After installation, make sure to be installed by running ```nvidia-smi``` command.)

```
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
sudo sh cuda_11.0.2_450.51.05_linux.run
```

Install cuDNN 8.0.<br>
To install cuDNN, download the corresponding version of packages (runtime, dev, and docs) from [NVIDIA developer site](https://developer.nvidia.com/cudnn) and install these packages as follows.

```
sudo dpkg -i libcudnn8_8.0.5.39-1+cuda11.0_amd64.deb
sudo dpkg -i libcudnn8-dev_8.0.5.39-1+cuda11.0_amd64.deb
sudo dpkg -i libcudnn8-samples_8.0.5.39-1+cuda11.0_amd64.deb
```

Setup is all done.

In the following instructions, make sure to install ```tensorflow-gpu==2.4.1``` instead of ```tensorflow==2.4.1```, and train by running command with ```--num_gpu``` option.
</blockquote>

## 2. Download and build Malmo ##

To install Malmo, you can use pre-built binary or build Malmo from source code.<br>
Here we download source code and build Malmo in Ubuntu 20.04.

First install **Python 3.6**, because Malmo is compatible with Python version 3.6.

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```

By running the following command, set Python 3.6 as default version for ```python3``` command.

```
# add python3.6 in update-alternatives
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
# configure python3.6 as default
sudo update-alternatives --config python3
```

> Note : This is needed, because Malmo's compilation runs ```python3``` command in cmake in order to check version for finding boost python module. (See ```CMakeLists.txt``` file in Malmo's source files.)

Check whether Python 3.6 is used in ```python3``` command.

```
python3 -V
```

Next, install and set up the required components for Malmo as follows.

```
# install required components
sudo apt-get install \
  build-essential \
  libpython3.6-dev \
  openjdk-8-jdk \
  swig \
  doxygen \
  xsltproc \
  ffmpeg \
  python-tk \
  python-imaging-tk \
  zlib1g-dev

# set environment for Java
echo -e "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> ~/.bashrc
source ~/.bashrc

# update certificates
sudo update-ca-certificates -f
```

> Note : Malmo uses the modded Minecraft and it then depends on Java SDK.

Download and build cmake as follows.

```
mkdir ~/cmake
cd ~/cmake
wget https://cmake.org/files/v3.11/cmake-3.11.0.tar.gz
tar xvf cmake-3.11.0.tar.gz
cd cmake-3.11.0
./bootstrap
make -j4
sudo make install
cd
```

Download and build Boost as follows.

```
mkdir ~/boost
cd ~/boost
wget http://sourceforge.net/projects/boost/files/boost/1.66.0/boost_1_66_0.tar.gz
tar xvf boost_1_66_0.tar.gz
cd boost_1_66_0
./bootstrap.sh --with-python=/usr/bin/python3.6 --prefix=.
./b2 link=static cxxflags=-fPIC install
cd
```

Now it's all ready for installing Malmo.<br>
Download and build Malmo as follows.

The generated file ```./install/Python_Examples/MalmoPython.so``` is then the entry point for Python package.

```
git clone https://github.com/Microsoft/malmo.git ~/MalmoPlatform
wget https://raw.githubusercontent.com/bitfehler/xs3p/1b71310dd1e8b9e4087cf6120856c5f701bd336b/xs3p.xsl -P ~/MalmoPlatform/Schemas
echo -e "export MALMO_XSD_PATH=$PWD/MalmoPlatform/Schemas" >> ~/.bashrc
source ~/.bashrc
cd ~/MalmoPlatform
mkdir build
cd build
cmake -DBoost_INCLUDE_DIR=/home/$USER/boost/boost_1_66_0/include -DBOOST_PYTHON_NAME=python3 -DCMAKE_BUILD_TYPE=Release ..
make install
cd
```

After Malmo compilation has completed, configure Python 3.8 as default for python3, because packages in Ubuntu 20.04 depends on Python version 3.8.

```
# add python3.8 in update-alternatives
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
# configure python3.8 as default
sudo update-alternatives --config python3
```

Check whether Python version 3.8 is set as default in python3.

```
python3 -V
```

> Note : See [here](https://github.com/microsoft/malmo/blob/master/doc/build_linux.md) for details about building Malmo.

## 3. Install required packages ##

Install required packages with dependencies (such as, TensorFlow, Ray framework with RLlib, etc) as follows.<br>
In this example, I have used TensorFlow for RLlib backend, but you can also use PyTorch for running RLlib.

First set up PIP in python3.6.

```
sudo apt-get update
sudo apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip
sudo apt-get install python3.6-distutils
```

Now let's install python packages in python 3.6.

```
python3.6 -m pip install \
  gym==0.21.0 \
  lxml \
  numpy \
  pillow \
  tensorflow==2.4.1 \
  gpustat==0.6.0 \
  ray[default]==1.6.0 \
  dm-tree==0.1.7 \
  attrs==19.1.0 \
  pandas

python3.6 -m pip install \
  ray[rllib]==1.6.0 \
  ray[tune]==1.6.0
```

> Note : When you run on GPU, install ```tensorflow-gpu==2.4.1``` instead of ```tensorflow==2.4.1```.

## 4. Configure desktop environment ##

Malmo is built on the modded Minecraft.<br>
It then needs monitor-attached environment, and here I configure X remote desktop environment and RDP service as follows.

```
sudo apt-get update
# while installation, select gdm3 for default display manager
sudo apt-get -y install xfce4
sudo apt-get -y install xrdp
sudo systemctl enable xrdp
echo xfce4-session >~/.xsession
sudo service xrdp restart
```

> Note : Run ```echo xfce4-session >~/.xsession``` for all users who runs the program.

Allow (Open) inbound port 3389 (which is default RDP port's number) in network settings to enable your client to connect to your server.

> Note : When you want to join into the same game with your own Minecraft client remotely, please open Minecraft port 25565 too.

## 5. Run Project Malmo with Minecraft ##

Login Ubuntu using remote desktop client.<br>
And run the following commands **on monitor-attached shell** (such as, LXTerminal) to launch Minecraft with Project Malmo's mod.

For the first time to run, all dependencies (including Project Malmo mod) are built and installed, and it will then take a while to start. (Please be patient to wait.)

```
cd MalmoPlatform/Minecraft
./launchClient.sh -port 9000
```

> Note : When you have troubles (errors) for downloading resources in minecraft compilation, please download [here](https://1drv.ms/u/s!AuopXnMb-AqcgdZkjmtSVg3VQL5TEQ?e=w4M4r7) and run the following command to use successful gradle cache.<br>
> ```mv ~/.gradle/caches/minecraft ~/.gradle/caches/minecraft-org```<br>
> ```unzip gradle_caches_minecraft.zip -d ~/.gradle/caches```<br>
> For troubles to use the monitor, see "Trouble Shooting" in the appendix below.

Keep running Minecraft with Malmo.

## 6. Train an agent (Deep Reinforcement Learning) ##

Now let's start training.<br>
Before starting, make sure that Minecraft with Malmo is running and listening port 9000. (See above.)

First, clone this repository.

```
git clone https://github.com/tsmatz/minecraft-rl-example
cd minecraft-rl-example
```

Copy file ```~/MalmoPlatform/build/install/Python_Examples/MalmoPython.so``` in current folder as follows.

```
cp ~/MalmoPlatform/build/install/Python_Examples/MalmoPython.so .
```

Run the training script (```train.py```) as follows.<br>
Note that this command is not needed to be run on the monitor attached shell. (This process will connect to Malmo instance running with a monitor and port 9000.)

```
python3.6 train.py /YOUR_HOME_DIR/minecraft-rl-example/lava_maze_malmo.xml
```

<blockquote>
Note : When you run on GPU, specify --num_gpus option as follows.

```
python3.6 train.py /YOUR_HOME_DIR/minecraft-rl-example/lava_maze_malmo.xml --num_gpus 1
```
</blockquote>

When you start the training code (```train.py```), you will see the running agent's view in 84 x 84 Minecraft's screen. This frame pixels are used by agent to learn.<br>
This frame size (84 x 84 x channel size) is supported for RLlib built-in convolutional network (ConvNet), and no custom model is then needed in this code. (Otherwise, create your own model and configure to use the custom model.)<br>
See the source code [visionnet.py](https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet.py) for the RLlib built-in convolutions.

This training will take around a day and a half long for completion, when running on a single GPU.

> Note : You can also run training on multiple workers in Ray cluster to speed up training. In the cluster, each workers should be configured to use a virtual monitor, because it will run as a batch in backgroud.<br>
> See [here](https://github.com/tsmatz/minecraft-rl-on-ray-cluster) for running this example on Ray cluster.

## 7. Run pre-trained agent

This repository also includes pre-trained checkpoint (```checkpoint/checkpoint-XXX``` in this repo) and you can then check the result soon.

After launching Minecraft with malmo port 9000 (see above), run the following command to run the pre-trained agent.

```
python3.6 run_agent.py /YOUR_HOME_DIR/minecraft-rl-example/lava_maze_malmo.xml
```

If you have your own trained checkpoint, you can also run and trace your own agent as follows.

```
python3.6 run_agent.py /YOUR_HOME_DIR/minecraft-rl-example/lava_maze_malmo.xml --checkpoint_file YOUR_OWN_CHECKPOINT_FILE_PATH
```

![Trace a trained agent](https://tsmatz.files.wordpress.com/2020/07/20200717_rollout_capture.gif)

## Appendix : Troubleshooting

**Xrdp won't accept a special character for password.**

Please create a new user with a simple password.

**Errors in Minecraft compilation or run**

If the download for Minecraft assets fails or cannot be found in Minecraft compilation, please download [here](https://1drv.ms/u/s!AuopXnMb-AqcgdZkjmtSVg3VQL5TEQ?e=w4M4r7) and use the successful cache as follows.

```
mv ~/.gradle/caches/minecraft ~/.gradle/caches/minecraft-org
unzip gradle_caches_minecraft.zip -d ~/.gradle/caches
```

**Error for connecting to Minecraft client**

When you have an error "Failed to find an available client for this mission" in ```startMission()```, please check as follows.

- Minecraft client is correctly running
- Not setting ```LC_ALL``` environment variable as follows (This will prevent from connecting)<br>
```export LC_ALL=C.UTF-8```

**Error in desktop session start**

See error details in ```~/.xsession-errors```, when it has some toubles to start xrdp (X remote desktop) session. Set ```mate-session``` in ```~/.xsession``` to fix, if needed.

**Azure DSVM or ML compute**

When you use data science virtual machine (DSVM) or [AML](https://tsmatz.wordpress.com/2018/11/20/azure-machine-learning-services/) compute in Azure :

- Deactivate conda environment, since MineRL cannot be installed with conda.

```
echo -e "conda deactivate" >> ~/.bashrc
source ~/.bashrc
```

- It will include NVidia cuda, even when you run on CPU VM. This will cause a driver error ("no OpenGL context found in the current thread") when you run Minecraft java server with malmo mod.<br>
  Thereby, please ensure to uninstall cuda.

```
sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```

**Errors for display setting (monitor)**

When your application cannot detect your display (monitor), please ensure to set ```DISPLAY``` as follows.<br>
(The error message "MineRL could not detect a X Server, Monitor, or Virtual Monitor" will show up.)

```
# check your display id
ps -aux | grep vnc
# set display id (when your display id is 10)
export DISPLAY=:10
```

When you cannot directly show outputs in your physical monitor, please divert outputs through a virtual monitor (xvfb).<br>
For instance, the following will show outputs (Minecraft game) on your own VNC viewer window through a virtual monitor (xvfb).

```
# install components
sudo apt-get install xvfb
sudo apt-get install x11vnc
sudo apt-get install xtightvncviewer
# generate xvfb monitor (99) and bypass to real monitor (10)
/usr/bin/Xvfb :99 -screen 0 768x1024x24 &
/usr/bin/x11vnc -rfbport 5902 -forever -display :99 &
DISPLAY=:10 /usr/bin/vncviewer localhost:5902 &
# run program
export DISPLAY=:99
python3.6 train.py
```
