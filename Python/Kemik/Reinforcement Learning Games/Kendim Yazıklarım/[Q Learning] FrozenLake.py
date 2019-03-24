# -*- coding: utf-8 -*-

import gym
import numpy as np
import random 
import matplotlib as plt

env = gym.make("FrozenLake-v0").env

# Q table oluşturuyoruz 
"""Biraz Oyun hakkında  bilgi vereyim 
Oyundaki amac Start (S) noktasından Goal(G) noktasına gidebilmek.Bu yolda F ler frozen path yani donmuş gidelibilen bir yol
fakat H'ler ise Hole yani boşluk anlamında. Eğer sonuca ulaşırsan +1 reward alırsın
4 adet actionımız var sağ sol yukarı aşağı
bu durumda 16 adet state'imiz olabilir. 4*4 lük bir harita için
"""
numberOfStates = env.observation_space.n # State sayısını env den aldık 
numberOfAction = env.action_space.n # Action sayısını envden aldık
q_Table= np.zeros([numberOfStates,numberOfAction])
#Hper Parametreler

alpha = 0.1 # Learning rate
gamma = 0.9 # Discount Faktor
epsilon = 0.1 # Epsilon for Exploit vs explore
# Görselleştirme için kullanılan metrix
reward_List = [] # Tamamen sonucları görsel üzerinde görmek için kulalandığımız liste
dropout_List = [] # Tamamen sonucları görsel üzerinde görmek için kulalandığımız liste


episode_Number= 10000 # 10000 kez eğit
for i in range (1,episode_Number):
    # initalize Enviroment
    state = env.reset()
    reward_Count =  0 # Rewardleri tutmak için kullandığımız indexi 0 ladık
    dropout_Count = 0 # Hatalı bırakmaları tutmak için kullandığımız indexi 0 ladık
    while True:
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
        if reward == -10 :  # Eğer reward -10 ise ben dropout yani taksiden yolcuyu yanlış yerde indirdim demektir.Bu bilgiyi oyundan biliyoruz
            dropout_Count+=1;
            
        if done:
            break
        reward_Count += reward
        dropout_List.append(dropout_Count)
        reward_List.append(reward_Count)
        if i%10 == 0 : # Çok fazla yazmasın diye yaptık 
            print("Episode :{},reward {},wrong dropput {}".format(i,reward_Count,dropout_Count))
            
# %% Sonucları TABLO HALİNDE görüntüleme
fig,axs = plt.subplots(1,2) # 1 satırda 2 tane plot oluştur
axs[0].plot(reward_List)
axs[0].set_xlabel("Bolum numarası")
axs[0].set_ylabel("Odulumuz")

axs[1].plot(dropout_List)
axs[1].set_xlabel("Bolum numarası")
axs[1].set_ylabel("Musteriyi hatalı yere bırakma sayısı")
axs[0].grid(True)
axs[1].grid(True)
plt.show()
