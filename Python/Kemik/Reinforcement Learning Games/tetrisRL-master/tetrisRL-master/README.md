![TETRIS RL](https://github.com/jaybutera/tetris-environment/blob/master/tetrisRL_logo.png)

[![PyPI
version](https://badge.fury.io/py/tetrisrl.svg)](https://badge.fury.io/py/tetrisrl)

## Installation
You need to have [pytorch](http://pytorch.org/) pre-installed. Easy to use
download scripts can be found on their website.

```bash
$ git clone https://github.com/jaybutera/tetrisRL
$ cd tetrisRL
$ python setup.py install
```
or
```bash
$ pip install tetrisrl
```

## Layout
* dqn_agent.py - DQN reinforcement learning agent trains on tetris
* supervised_agent.py - The same convolutional model as DQN trains on a dataset of user playthroughs
* user_engine.py - Play tetris and accumulate information as a training set
* run_model.py - Evaluate a saved agent model on a visual game of tetris (i.e.)
```bash
$ python run_model.py checkpoint.pth.tar
```

## Usage

### Using the Environment
The interface is similar to an [OpenAI Gym](https://gym.openai.com/docs) environment. 

Initialize the Tetris RL environment

```python
from engine import TetrisEngine

width, height = 10, 20
env = TetrisEngine(width, height)
```

Simulation loop
```python
# Reset the environment
obs = env.clear()

while True:
    # Get an action from a theoretical AI agent
    action = agent(obs)

    # Sim step takes action and returns results
    obs, reward, done = env.step(action)

    # Done when game is lost
    if done:
        break
```

## Example Usages

### Play Tetris for Training Data
Play games and accumulate a data set for a supervised learning algorithm to
trian on. An element of data stores a
(state, reward, done, action) tuple for each frame of the game.

You may notice the rules are slightly different than normal Tetris.
Specifically, each action you take will result in a corresponding soft drop
This is how the AI will play and therefore how the training data must be taken.

To play Tetris:
```bash
$ python user_engine.py
```

Controls:  
W: Hard drop (piece falls to the bottom)  
A: Shift left  
S: Soft drop (piece falls one tile)  
D: Shift right  
Q: Rotate left  
E: Rotate right  

At the end of each game, choose whether you want to store the information of
that game in the data set. Data accumulates in a local file called 'training_data.npy'.

### Example supervised learning agent from data
Run the supervised agent file and specify the standard training data file generated in the previous step as a command
line argument.
```bash
$ python supervised_agent.py training_data.npy
```

### Example reinforcement learning agent
```bash
# Start from a new randomized dqn agent
$ python dqn_agent.py
# Start from a the last recorded dqn checkpoint
$ python dqn_agent.py resume
# Specify a custom checkpoint
$ python dqn_agent.py resume supervised_checkpoint.pth.tar
```

The DQN agent currently optimizes on a metric of freedom of action. In essence the agent should learn to maximize
the entropy of the board. A player in Tetris has the most freedom of action when the area is clear of pieces.

### Watch a checkpoint play a game
```bash
$ python run_model.py checkpoint.pth.tar
```
