# RLOptimalTradeExecution


Installation steps:

git clone https://github.zhaw.ch/dasc/RLOptimalTradeExecution.git
cd RLOptimalTradeExecution
source venv/bin/activate
pip install -r requirements.txt



In order to use the visualisation, set SHOW_UI=True in ppo_sbaseline.py file. After the agent was trained/loaded, it will go in "inference" mode. To control the stepping through the environment, bring the app UserInterface into focus and use the following keys on your keyboard:

S = is for toggling the pause/resume between Steps
N = while you are in "S" pause mode, the N key can be used to move forward in time 1 step.


E = is for toggling the pause/resume between Episodes (for the moment there is a little bug here, ignore E)
