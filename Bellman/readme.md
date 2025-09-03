Implementação de um algoritmo chamado Iteração de Valor para um ambiente simples de Grid World. Este algoritmo aplica iterativamente a Equação de Bellman para encontrar a função de valor ótima.

Ambiente: Grid World 4x4

Um grid de 16 estados (0 a 15).

Estado Terminal (Meta): Estado 15 (canto inferior direito) com recompensa +1.

Estado Terminal (Penalidade): Estado 10 com recompensa -1.

Ações: Cima, Baixo, Esquerda, Direita.

Política 
pi: Uma política aleatória simples, onde cada ação tem 25% de chance de ser escolhida.

Dinâmica p: Determinística. Se você escolher "Direita", você se move para a direita (a menos que haja uma parede).

Fator de Desconto 
gamma: 0.9.

É calculado iterativamente a função de valor (V_pi(s)) para cada estado. 
É mostrado os valores "se propagando" a partir dos estados terminais, até que todo o mapa de valores se estabilize.

![](https://raw.githubusercontent.com/renansaraivaifpb/fundamentals-of-reinforcement-learning/refs/heads/main/Bellman/value_function_iteration_1.png)
![](https://raw.githubusercontent.com/renansaraivaifpb/fundamentals-of-reinforcement-learning/refs/heads/main/Bellman/value_function_iteration_2.png)
![](https://raw.githubusercontent.com/renansaraivaifpb/fundamentals-of-reinforcement-learning/refs/heads/main/Bellman/value_function_iteration_3.png)
![](https://raw.githubusercontent.com/renansaraivaifpb/fundamentals-of-reinforcement-learning/refs/heads/main/Bellman/value_function_iteration_6.png)
![](https://raw.githubusercontent.com/renansaraivaifpb/fundamentals-of-reinforcement-learning/refs/heads/main/Bellman/value_function_iteration_11.png)
![](https://raw.githubusercontent.com/renansaraivaifpb/fundamentals-of-reinforcement-learning/refs/heads/main/Bellman/value_function_iteration_Final.png)
