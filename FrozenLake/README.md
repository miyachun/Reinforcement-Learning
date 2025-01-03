https://gymnasium.farama.org/environments/toy_text/frozen_lake/  

---
(DQN)Deep Q Network  
1)經驗回放緩沖區(Experience Replay Buffer)  
收集經驗資料於Main network，當資料足夠時，抽樣進行訓練  

2)損失函數(Loss Function)  
MSE（Mean-Square Error）  

$$ L(\theta)= \frac {1}{N} \sum_{i=1}^N \left( ri+\ \gamma \underbrace{max} _{a^1} Q { _\theta }^1 (S _i ^1,a^1)-Q _\theta(S_i,a_i)\right)^2  $$  

(Target network)Target Q value  

$$ ri+\ \gamma \underbrace{max} _{a^1} Q { _\theta }^1  (S _i ^1,a^1) $$  

(Main network)Predicted Q value  

$$ Q _\theta(S_i,a_i)  $$  

3)最佳化法Optimizer  
梯度下降法(Gradient Descent)  

$$ \theta=\theta-a \nabla_\theta L(\theta) $$  

學習速率  $$a  $$  
梯度  $$\nabla_\theta L(\theta) $$

---

(DDQN)Double DQN Network  

$$ L(\theta)= \frac {1}{N} \sum_{i=1}^N \left( ri+\ \gamma Q { _\theta }^1 (S^1 , \underbrace{argmax} _{a^1} Q { _\theta } (S_i^1,a^1)) -Q _\theta(S_i,a_i)\right)^2  $$  


---

MSE（Mean-Square Error）  
均方誤差（MSE）度量的是預測值和實際觀測值間差的平方的均值。它只考慮誤差的平均大小，不考慮其方向。但由於經過平方，與真實值偏離較多的預測值會比偏離較少的預測值受到更為嚴重的懲罰。再加上 MSE 的數學特性很好，這使得計算梯度變得更容易。均方誤差（Mean-Square Error, MSE）在維基百科上的解釋：是對於無法觀察的參數的一個估計函數T；其定義為：  
  
$$ MSE(T)=E((T-\theta)^2) $$  

即，它是「誤差」的平方的期望值。誤差就是估計值與被估計量的差。均方誤差滿足等式：  

$$ MSE(T)=var(T)+(bias(T))^2 $$  

其中  

 $$ bias(T)=E(T)-\theta $$    
 
也就是說，偏差是估計函數的期望值與那個無法觀察的參數的差。  

均方誤差（MSE）對於損失函數的意義:  
均方誤差 (MSE) 是最常用的回歸損失函數，計算方法是求預測值與真實值之間距離的平方和，公式如下：  


$$ MSE= \sum_{i=1}^n \left(yi-yi^p \right)^2  $$  

  
