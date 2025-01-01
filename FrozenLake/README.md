https://gymnasium.farama.org/environments/toy_text/frozen_lake/  


MSE（Mean-Square Error）  

$$ L(\theta)= \frac {1}{N} \sum_{i=1}^N \left( ri+\ \gamma \underbrace{max \ Q{_\theta}^1} _{a^1}  (S _i ^1,a^1)-Q _\theta(S_i,a_i)\right)^2  $$  

Target Q value  
$$ ri+\ \gamma \underbrace{max \ Q{_\theta}^1} _{a^1}  (S _i ^1,a^1) $$  
Predicted Q value  
$$ Q _\theta(S_i,a_i)  $$  
