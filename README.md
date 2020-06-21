A comparison of deep reinforcement learning techniques using a method that can be run on simple laptop hardware.

Using AIgym: Cart-Pole-V0  environment

**Algorithms and Motivations**


**DQN algorithm**
The simplest solutions was the DQN algorithm, which uses neural networks as function 4​approximators​ . In this model two networks are used, one during the learning and one for determining the best actions for the agent to take. An experience replay buffer is used which stores the raw experiences the agent has had when interacting with the environment and separates out the learning from the acting. It is thought that doing so gives a better representation of the true data as the assumption of independence is maintained, as well as attempting to provide a solution to the issue of gradient descent on incomplete knowledge (so not true gradient descent). As a result actions taken by the first neural network are still online, i.e. interacting with the environment and updating, however the second neural network is updated less regularly at a time step determined by the programmer.


**Double DQN algorithm**
Whilst deep Q networks are effective and will be evaluated in this experiment they are frequently seen to overestimate the maximum expected value for a given action. One method5​for overcoming this is double deep Q learning (DDQN) ​ , which utilises the same two networkarchitecture seen in DQN, implementing them in a slightly different manner however. DQN model (and Q learning) updates are based on parameters from a network, which also receives updates from the same network, i.e. the Bellman equation is dependent on itself. This means that when the model is initialized it will be taking the max state of random initialized values which it does not have accurate representations of. DDQN instead used the local network to select the action for the next state, with the target network providing the Q values. The target network Q values are more stable as learning is not online.


**Dueling Double DQN Algorithm**
A final model type examined in this experiment is the dueling double DQN model, seen as an 6​advancement of the previous double DQN model​ , due to its ability to again address the overestimation of actions. In addition it has been shown to improve the training instability seen with DDQN models which result from a single network picking the actions (the moving target problem). The architecture used in the dueling double DQN to achieve this is distinct from the previous two examples. In this model they separate out the state values and the advantage function (a new addition) within a single network before recombining the output. The advantage function is the novelty here which calculates the advantage of being in any state, with the intention of identifying states that will not be useful and do not need to be focused on. Actions can then be selected from either the Q table or this aggregated advantage function.


**Quantitative Analysis**


The official definition of solving Cartpool-v0 is to obtain 100 epochs with a score over 195. To enable algorithms to run in a timely manner a comparison of achieving a score of over 195 for 10 epochs, with a forced stopping at 2000 epochs was conducted. This limitation potentially does not accurately evaluate stable learning, however this was accepted as comparison was still possible over the iterations.

![image](https://user-images.githubusercontent.com/52289894/85226826-6f982480-b3d1-11ea-8b7f-b13ebec20d32.png)

The double DQN algorithm was able to solve the environment in 1870 epochs whilst the single DQN algorithm was not able to solve the environment in under 2000 epochs. Generally both algorithms performance increases over time as more states are explored, hence more experiences are accumulated in their replay memory and the quality of the experiences in the replay memory improves. It was also observed that the curves for all the algorithms become less noisy as the algorithms make transition from exploration to exploitation mode with the decrease in epsilon decay rate. The exception here is double DQN in later episodes where stability appeared to break down, contrary to our hypothesis, the reason for this is not attributed to the small averaging period and potential noise.
Both algorithms had similar learning rates for the first 100 iterations, this is the stage when both algorithms were in exploration mode and the quality of experience in their replay memory was not good. Post iteration 110; the learning rate of double DQN appeared to be more stable with an upward trend when compared to DQN which had higher oscillation with near constant average value.


A second test was completed using another variant of double DQN, double duel DQN (DDDQN). This variant of the algorithm differs from DDQN by the structure of its network. The algorithm includes the trainable online model and with the separation of advantage and value layer that was not previously present. DDDQN enabled the model to learn the environment in a smaller number of iterations (1174) than both DQN and DDQN with a significantly more stable learning process.
![image](https://user-images.githubusercontent.com/52289894/85226836-79218c80-b3d1-11ea-88e8-6b4f65b3c7cb.png)

**Qualitatively Analysis**
The results in Figure 7 showed a clear difference in performance between DQN, DDQN and DDQN, which confirms our hypothesis. The higher predicted values and larger oscillation of scores throughout the training process for DQN. The lower score for DQN throughout the training process implies overestimation of Q values by its target network. This overestimation causes more abrupt changes in parameters of the main network during backpropagation. The abrupt changes in parameters value of the main network get transmitted to the target network, which makes the target network to have a very different prediction distribution for Q values after the updates in its parameters.
For DDQN, responsibility of selecting best actions is shared between main and target networks. The actions that both the main and target networks consider the best are given higher Q values than the actions that the networks do not agree on. This process of action selection tackles the issueof QvaluesoverestimationthattakesplacebyvanillaDQN,andislikelytobethereason for smaller oscillation in DDQN’s score curve earlier on than DQN’s.
DDDQN algorithm had the quickest convergence time and very stable training progression. This is likely due to the property of the algorithm that it calculates the quality being in a certain state without conditioning it to a particular action. This property of the algorithm is more beneficial for problems like ours in which taking action does not change the environment and only impact the states. Although the environment in this study was very simple, DDDQN and DDQN outperformed DQN, both, in terms of convergence time and robustness of predicted score per iteration. The performance of these two models becomes even more superior when applied in more complex environments where there is a higher variety of actions to choose from.


**Reference**
1. Sutton, R. S. & Barto, A. G. ​Reinforcement learning: an introduction.​ (The MIT Press, 2018).
2. Weber, R. On the Gittins Index for Multiarmed Bandits. ​Ann. Appl. Probab. 2​ ​, 1024–1033
(1992).
3. Kuleshov, V. & Precup, D. Algorithms for multi-armed bandit problems. ​ArXiv14026028 Cs
(2014).
4. Mnih, V. ​et al.​ Playing Atari with Deep Reinforcement Learning. ​ArXiv13125602 Cs​ (2013).
5. Hasselt, H. V. Double Q-learning. in ​Advances in Neural Information Processing Systems 23
(eds. Lafferty, J. D., Williams, C. K. I., Shawe-Taylor, J., Zemel, R. S. & Culotta, A.)
2613–2621 (Curran Associates, Inc., 2010).
6. Wang, Z. ​et al.​ Dueling Network Architectures for Deep Reinforcement Learning.
ArXiv151106581 Cs​ (2016).
7. van Hasselt, H., Guez, A. & Silver, D. Deep Reinforcement Learning with Double Q-learning.
ArXiv150906461 Cs​ (2015).
