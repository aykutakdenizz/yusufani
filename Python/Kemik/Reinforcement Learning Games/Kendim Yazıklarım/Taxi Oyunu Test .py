# -*- coding: utf-8 -*-
import gym # Oyunlarımızın Oldugu GYM kutuphanesini dahil ettik

env = gym.make("Taxi-v2").env # Taxi 2 oyununu ekledik
env.render() # Oyunu GUI olarak pencere halinde açan komut 
"""
blue = passenger
purple = destination
yellow/red = empty taxi
green = full taxi
RGBY = location for destination and passanger

"""
env.reset() # Oyun ortamımızı tekrardan başlatır

# %%

print("Toplam State sayımız",env.observation_space) # (5*5 adet karede) *(4 farklı adrese yolcuyu bırakabiliriz.) * ( 4 adet yerden veya oldugumuz yerden yolcu alabiliriz)
print("Action sayımız",env.action_space) # yolcu al , yolcu bırak , sağa git , sola git , yukarı git ,  aşağı git 

state= env.encode(3,1,2,3) # sırayla taxi row , tax, column , passenger index, destination
# state bize bir state numberı vericek
env.s=state
env.render()
# %%
"""
Actions: 
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
"""
# probability, next_state, reward, done=episodeun bitip bitmemesi
env.P[331]

# %%
total_reward_list = []
# episode 
for j in range(5):
    env.reset()
    time_step=0 # adım sayısı
    total_reward=0# Toplam ödül
    list_visualize = []
    while True:
        # choose action
        action = env.action_space.sample()
        #perform action and get reward
        state,reward,done,info=env.step(action) # Bize yeni state'i, ödülü, episodeun tamamalanıp tamamlanmadığını döndürdü
        #Total reward
        total_reward += reward
        #visualize
        #bilgileri 1 listede tutalım ki en son inceleyeim
        list_visualize.append({"frame" : env.render(mode = "ansi") ,
                           "state" : state , "action":action,"reward":reward,"Total_Reward":total_reward    
                           })
        #env.render()
        if done: # Eger episode  t amamsa donguden cık
            total_reward_list.append(total_reward)
            break
 # %% 
 #Listedeki verileri incelemek icin
import time
for i,frame in enumerate(list_visualize):
   print(frame["frame"])
   print("Timestate: ",i+1)
   print("state ",frame["state"])
   print("Action ",frame["action"])
   print("Reward ",frame["reward"])
   print("Total_Reward ",frame["Total_Reward"])
   # time.sleep(1) # 1 saniyelik aralıklarla yazdırsın 

