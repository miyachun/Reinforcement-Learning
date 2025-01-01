https://gymnasium.farama.org/environments/toy_text/frozen_lake/  


MSE（Mean-Square Error）  
均方誤差（MSE）度量的是預測值和實際觀測值間差的平方的均值。它只考慮誤差的平均大小，不考慮其方向。但由於經過平方，與真實值偏離較多的預測值會比偏離較少的預測值受到更為嚴重的懲罰。再加上 MSE 的數學特性很好，這使得計算梯度變得更容易。  

$$ L(\theta)= \frac {1}{N} \sum_{i=1}^N \left( ri+\ \gamma \underbrace{max \ Q{_\theta}^1} _{a^1}  (S _i ^1,a^1)-Q _\theta(S_i,a_i)\right)^2  $$  

Target Q value  

$$ ri+\ \gamma \underbrace{max \ Q{_\theta}^1} _{a^1}  (S _i ^1,a^1) $$  

Predicted Q value  

$$ Q _\theta(S_i,a_i)  $$  

