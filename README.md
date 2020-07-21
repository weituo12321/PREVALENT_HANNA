# PREVALENT_HANNA
Apply PREVALENT pretrained code on HANNA task


Environment setting up:
1 sudo bash scripts/build_docker.sh
Possible issues:
Step 1/17 : FROM nvidia/cudagl:10.0-devel-ubuntu18.04
Get https://registry-1.docker.io/v2/: dial tcp: lookup registry-1.docker.io on [::1]:53: read udp [::1]:39620->[::1]:53: read: connection refused

You need to log in your dockerhub account to pull images


2 set your environment
2.1 Export MATTERPORT_DATA_DIR=/home/victor/work/vlnwork/Matterport3DSimulator/skybox_images/v1/scans

2.2 create your container
sudo -E bash scripts/test_run_docker.sh

2.3 build simulator
bash scripts/build_simulator.sh

Install missing dependencies:
apt-get install python3-venv python3-pip
pip3 install pytorch_transformers==1.2.0  




2.4 test if simulator is ready
python scripts/test_api.py

2.5
modify define_vars.sh   # set your folder to store the result models
modify train_main.sh     #  set your gpu id and output folder name
modify config/hanna.json   # training parameter setting
cd tasks/HANNA/exp_scripts
bash train_main.sh

2.6 to test your saved model
bash eval_main.sh unseen_all 


