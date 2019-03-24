# -*- coding: utf-8 -*-

import gym
import numpy as np
import random 
import matplotlib.pyplot as plt
env = gym.make("Taxi-v2").env

# Q table oluşturuyoruz 
numberOfStates = env.observation_space.n # State sayısını env den aldık 
numberOfAction = env.action_space.n # Action sayısını envden aldık
q_Table= np.zeros([numberOfStates,numberOfAction])
#Hper Parametreler

alpha = 0.1 # Learning rate
gamma = 0.9 # Discount Faktor
epsilon = 0.1 # Epsilon for Exploit vs explore
# Görselleştirme için kullanılan metrix
reward_List = [] # Tamamen sonucları görsel üzerinde görmek için kulalandığımız liste
episode_Number= 10000 # 10000 kez eğit
for i in range (1,episode_Number):
    # initalize Enviroment
    state = env.reset()
    reward_Count =  0 # Rewardleri tutmak için kullandığımız indexi 0 ladık
    while True:
        env.render()
        # Exploit vs explore to find aciton
        # epsilon değerimiz % 10 ihtimalle yeni yol keşfedecek %90 arasında bilinen iyollardan devam et 
        if random.uniform(0,1) < epsilon : # Keşfet
            action = env.action_space.sample() # Herhangi bir actionu samplelar arasından seç 
        else:
            action = np.argmax(q_Table[state])
        
        # Action process and  take reward / observation 
        next_State,reward,done,info=env.step(action) # Bir adım sonraki bilgileri aldık
        
        # Q learning Function
        old_Value = q_Table[state,action]#Old value
        next_Max = np.max(q_Table[next_State]) # Next MAX
        next_Value = (1-alpha) * old_Value + alpha*(reward+gamma*next_Max) # Q fonksiyonunu yazdık
        
        # Q table update 
        q_Table[state,action] = next_Value
        # Update State
        state =next_State
        # Find wrong Droputs (Ne kadar yanlış yerde yolcu indirdiğimizn sayısı tuttuk)
        if done:
            break
        reward_Count += reward
        reward_List.append(reward_Count)
        if i%10 == 0 : # Çok fazla yazmasın diye yaptık 
            print("Episode :{},reward {}".format(i,reward_Count))
            
plt.plot(reward_List)
