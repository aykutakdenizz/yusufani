# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:17:07 2019

@author: Yusuf Anı
"""
import gym # Enviorenment için
import numpy as np 
from collections import deque # 
from keras.models import Sequential # Sequential model
from keras.layers import Dense # Dense Layer
from keras.optimizers import Adam # Optimizer 
import random
import matplotlib.pyplot as plt
""" 
Oyundan bahsedeyim 
Asıl amaç çubuğu 15 dereceden fazla eğmemek ve altındaki kartonu çok fazla kaydırmamak
Observationlarımız 
0-> Cartın pozisyonu
1-> Cartın hızı
2-> Çubuğun açısı
3-> Cubuğun karta göre hızı

Actionlarımız 
0-> cartı sola it
1-> cartı sağa it 

Ödülümüz 
Her bir adım başına 1 puan
"""

class DQLAgent :
    def __init__(self,env):
        #parameter and hyperparameter
        self.state_Size=env.observation_space.shape[0]#State size bizim neural networkdaki başlangıç statemizdeki nodelar olacak 
        self.action_Size= env.action_space.n #Action size da bizim neural networkdeki çıkıs nodelarımız
        
        self.gamma= 0.99 #Discount rate
        self.learning_rate= 0.001 
        self.epsilon = 1 # Action seçerkenki değer explore
        self.epsilon_decay=0.995 # Epsilonun her adımda azalma miktarı
        self.epsilon_min= 0.01  # Epsilonuın alabileceği minimun değer
       
        self.memory = deque ( maxlen= 1000) # Deque 1000 lik bir liste gibi düşün
        
        self.model = self.build_model() # Gerekli olan neural networkü oluşturan fonksiyon
    def build_model(self):
        #Neural network build for q learningg
        model = Sequential()  # Neural yapısını oluşturduk 
        model.add(Dense(48,input_dim= self.state_Size, activation = "tanh")) # 48 adet nöronlu State_size kadar nodelu tanh activation kullanan bir layer oluşturuduk
        model.add(Dense(self.action_Size,activation="linear")) # action size kadar çıkış nodu olan linear aktivasyonu kullanan layer
        model.compile(loss= "mse", optimizer = Adam(lr = self.learning_rate)) # mse loss fonksiyonunu kullanan ADAM optimizerını kullanan bir yapı
        return model
    def remember(self,state,action,reward,next_State,done):
        #storage Elimizdeki değerleri hafızaya atan fonksiyon
        self.memory.append((state,action,reward,next_State,done))
    def act(self,state):
        #acting : explore an exploit
        if random.uniform(0,1) <=self.epsilon: # Explore
            return env.action_space.sample()
        else: # exploit
            act_values = self.model.predict(state) # Bana modeldeli en yüksek değerli actionun indexini döndürmesi lazım o state e göre
            return np.argmax(act_values[0])
        
    def replay(self,batch_Size):
        #training
        if len(self.memory) < batch_Size:
            return # Yani elimde batch size kadar state yoksa bir şey yapmadan dön
        mini_batch=random.sample(self.memory,batch_Size) # Rastgele olarak batch size kadar örnek al sampledan
        for state,action,reward,next_State,done in mini_batch: 
            if done: # Eğer oyun bitmişsse herhangi bir state durumuna bakmaya gerek yok direkt ödülümüz sonuç olarak çıkacaktır
                target = reward
            else:
                #Target değeri burada aslında bizim gerçek değerimiz oluyor.Loss fonksiyonunu yazabilmek için bu değer lazım
                target=reward + self.gamma*np.amax(self.model.predict(next_State)[0]) # A max tüm değerleri tek bir liste haline getiriyor bu yüzden [0] a baktık
                #Formüle göre qlearning vs deep q learining dosyasında var kırmızı işaretli alan yeni statedeki en büyük aksiyonun i
            train_Target = self.model.predict(state) # Kendi statemin sonucu
            train_Target[0][action] = target  # %100 EMİN OLMAMAKLA BERABER
            """ Aslında burada da bir loss hesaplıyoruz. Gerçekte olması gereken değer dediğimiz targetı en yüksek eleandaki yere koyuyoruz."""
            self.model.fit(state,train_Target,verbose = 0 ) # Verbose bazı infoları yazmamaıs için 
            # Training işlemini fitfonksiyonu ile gerçekleştirdik.
    def apativeEGreedy(self):
        #ilk başlarda çok fazla keşfetmek sonra ağırlıklarımıza(statelerimize) güvenmemiz gerekiyor
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
if __name__ == "__main__":
    #initalize env and agent
    env = gym.make("CartPole-v0") # OYunu env a yükledik 
    agent = DQLAgent(env) # Agentımızı oluşturduk
    batch_Size = 64 # Yani storageden 16 adet parametre kullanıcaz.Yani state,action,reward,next_State , done ların hepsinden 16 adet kullanıcaz
    episodes=500 # Tekrar sayisi 
    reward_List = []
    for e in range (episodes): 
        # initalize environment
        state= env.reset()
        state = np.reshape(state,[1,4]) # State bilgilerinin hepsini 1 elemanda toplamak ilerisi için bize yardımcı olacak .En son stateleri bir listede tutarken mesela
        time = 0 # Geçen zamanı tutmak için
        total_reward=0
        while True :
            #act
            action = agent.act(state) # Hangi actionı almamız gerektiğini belirledik
            #step
            next_State,reward,done,info=env.step(action) # Actionı çalıştırınca yeni state ve rward değerlerini aldık
            next_State = np.reshape(next_State,[1,4]) # State bilgilerinin hepsini 1 elemanda toplamak ilerisi için bize yardımcı olacak .En son stateleri bir listede tutarken mesela
            #remember / Storage
            agent.remember(state,action,reward,next_State,done)
            #update state
            total_reward+=reward
            state= next_State
            #replay
            agent.replay(batch_Size)
            #adjust epsilon
            agent.apativeEGreedy()
            time += 1 
            if done:
                print("Bölüm {}, time : {}".format(e,time))
                reward_List.append(total_reward)
                break
    plt.plot(reward_List) # 1 satırda 2 tane plot oluştur
    plt.show()
#selection= input("Modeli kaydetmek için 1 e basın ")
#if selection == 1:
#    agent.model.save('model.tf')
#    print("Model Kaydedildi")
# %% görselleştirme kısmı / Test bölümü
#import  time
#from keras.models import load_model
#selection = input("Daha onceki modelleri kullanmak istiyorsanız 1 yazınız ")
#if selection== '1' : 
#    trained_model =load_model("model.tf")     
#else :
#    trained_model = agent
#state = env.reset()
#state = np.reshape(state, [1,4])
#time_t = 0
#while True :
#    env.render()
#    action=trained_model.act(state)
#    next_state,reward,done,info = env.step(action)
#    next_state = np.reshape(next_state, [1,4])
#    state = next_state
#    time_t+=1
#    print(time_t)
#    time.sleep(0.4)
#    if done:
#        break
#print("Test İşlemi başarıyla sonlandırıldı")
                