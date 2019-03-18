
import gym
import numpy as np
import time, pickle, os

env = gym.make('FrozenLake-v0')

epsilon = 0.9
# min_epsilon = 0.1
# max_epsilon = 1.0
# decay_rate = 0.01

total_episodes = 10000
max_steps = 100

lr_rate = 0.81
gamma = 0.96 # Discount faktorumuz

Q = np.zeros((env.observation_space.n, env.action_space.n)) # Q tablemiza 16 statelik 4 actionlık  yer açtık
    
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)
#Oyunun amacı (S)tart poziyonundan (G) pozisyonuna gitmek ve agent (F) de durabilirken (H) de duramaz
# 4 temel aksiyon var yukarı asagı sag sol 
# Her F statinde 0 ödül alır her H stateinde -1 ödül alır Her G stateinde ise +1 ödül alır
# Start
rewards=0

for episode in range(total_episodes):
    state = env.reset() # Suanki durum icin genel bilgileri state'e attık
    t = 0  # t bizim icin step sayısı
    while t < max_steps:
        env.render() # Tabloyu burada yazdırıyor
        action = choose_action(state)  # random degerimiz epsilon degerinden kücükse rastgele bir aksiyon , degilse q tabledaki en yüksek olasılıklı aksiyon
        

        state2, reward, done, info = env.step(action)  # secilen aksiyonun sonucları 

        learn(state, state2, reward, action)  # Q tableimiz icin gerekli hesaplamaları yaptırttıgımız fonksiyon

        state = state2

        t += 1
        rewards+=1
        if done:
            break
#     epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode) 
        # os.system('clear')
        time.sleep(0.1)

    
print ("Score over time: ", rewards/total_episodes)
print(Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)








