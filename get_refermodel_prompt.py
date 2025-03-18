# flake8: noqa

abstract_compare_prompt_cn = """##人物简介
你是一个根据modelA的abstract以及相关models的abstract，对相关models中的各个model改进到modelA的难度进行排序的专家

##modelA的abstract
{modelA}

##相关models的abstract
{cfg}

##回复格式，请以json文件格式回复，类似一种二维字典：
{{
"模型名称1"：{{"模型名称","原因"}},
"模型名称2"：{{"模型名称","原因"}},
"模型名称3"：{{"模型名称","原因"}},
}}

##样例
###输入

###输出

##注意
1.回复的字典中的模型顺序是从最容易实现modelA到最难实现modelA，原因中填写其容易实现或难实现modelA的原因
2.想好答案之后，再自我检查一遍，确认无误后返回
"""

refermodel_prompt_cn = ("""##人物简介
你是一个根据论文中\"Experiments\"及\"References\"部分提取出该论文用作对比实验的模型及其出处的专家

##论文的Experiments
{Experiments}

##论文的References
{References}

##回复格式，请以json文件格式回复，类似一种二维字典：
{{
"模型名称1"：{{"模型名称","论文名称":,"Arxiv地址"}},
"模型名称2"：{{"模型名称","论文名称":,"Arxiv地址"}},
"模型名称3"：{{"模型名称","论文名称":,"Arxiv地址"}},
}}

##样例

###输入

# 5 Experiments  

In this section, we assess the performance of the proposed DM and HDM models, comparing them with Transformer-. based methods, DT and HDT. Our evaluation spans over seven distinct tasks sourced from the D4RL benchmark, selected precisely for its provision of a diverse set of tasks, each accompanied by multiple demonstration data sets of varying quality. As the efficacy of these methods relies heavily on their capacity to approximate policies represented in. the demonstration data set, the resulting performance of the methods is intrinsically linked to the quality of these data sets. Consequently, we conduct our training procedures using the diverse data sets provided by the benchmark for each. respective task.  

![](images/c200417e7d19f17a1e0826ec7e3df170c4399497eb2c316bba3cf410a0d8276e.jpg)  
Figure 4: Comparison of the performance of the DM varying architecture configuration across the 7 D4RL tasks, for different demonstration data sets. The scale of the bar graphs is the maximum reward present in the respective data set.. L is the number of layers, D is the embedding size, and K is the context length..  

![](images/c560bc2b34fdcd394bba3b1cc7a8dbc41b5e59e47a7751ff0e192ca15bb717d1.jpg)  
Figure 5: Comparison of the performance of the DM with the sequence of RTG, varying architecture configuration across the 7 D4RL tasks, for different demonstration data sets. The scale of the bar graphs is the maximum reward present in the respective data set. L is the number of layers, D is the embedding size, and K is the context length. The values are obtained by using the maximum reward of the data set as the desired reward.  

Table 1: Maximum accumulated returns of the DT, HDT, DM, DM with RTG, and HDM methods using 6 layers, an embedding size of 128 and context length of 20. We test the models on seven tasks from the D4RL [2] benchmark and vary the demonstration data sets. Highest values are highlighted in bold..   


<html><body><table><tr><td rowspan="2"></td><td rowspan="2">Data Set</td><td colspan="3">DT</td><td rowspan="2">HDT</td><td colspan="3">DM w/ R</td><td rowspan="2">DM wo/ R</td><td rowspan="2">HDM</td></tr><tr><td>Half</td><td>Max</td><td>10k</td><td>Half</td><td>Max</td><td>10k</td></tr><tr><td rowspan="3">Ant</td><td>expert</td><td>2722.84 ± 9.27</td><td>2731.52 ± 3.91</td><td>2733.84 ± 9.3</td><td>2731.83 ± 8.38</td><td>1853.28 ± 7.54</td><td>2735.34 ± 11.45</td><td>2742.30 ±1 14.37</td><td>2746.57 ± 10.69</td><td>2749.2 ± 11.62</td></tr><tr><td>medium</td><td>787.54 ± 22.87</td><td>801.41 ± 14.61</td><td>800.91 ± 17.32</td><td>787.23 ± 33.33</td><td>760.23 ± 21.73</td><td>782.97 ± 13.83</td><td>803.63 ± 14.76</td><td>821.03 ±18.21</td><td>820.39 ± 23.41</td></tr><tr><td>medium-expert</td><td>1525.61 ± 724.43</td><td>2718.68 ± 11.66</td><td>2713.55 ± 15.06</td><td>2171.27 ± 792.37</td><td>1769.70 ± 18.46</td><td>2742.88 ± 13.64</td><td>2734.04 ± 15.92</td><td>2738.41 ± 8.42</td><td>2730.01 ± 12.45</td></tr><tr><td rowspan="4">Antmaze</td><td>medium-replay</td><td>650.23 ± 63.67</td><td>732.89 ± 13.76</td><td>815.53±42.72</td><td>712.61 ± 39.5</td><td>639.22 ± 20.49</td><td>694.40 ± 13.69</td><td>716.67 ± 37.54</td><td>716.85 ± 33.65</td><td>728.2 ± 48.64</td></tr><tr><td>large-diverse</td><td>0.02 ± 0.04</td><td>0.05 ± 0.05</td><td>0.05 ± 0.05</td><td>0.02 ± 0.04</td><td>0.0 ± 0.01</td><td>0.0 ± 0.01</td><td>0.0 ± 0.01</td><td>0.08 ±0.13</td><td>0.05 ± 0.05</td></tr><tr><td>medium-diverse</td><td>0.2 ± 0.17</td><td>0.18 ± 0.19</td><td>0.2 ± 0.17</td><td>0.12 ± 0.04</td><td>0.0 ± 0.05</td><td>0.1 ± 0.05</td><td>0.5 ± 0.05</td><td>0.2 ± 0.17</td><td>0.15 ± 0.05</td></tr><tr><td>umaze</td><td>0.92 ± 0.04</td><td>0.92 ± 0.04</td><td>0.92 ± 0.04</td><td>0.52 ± 0.25</td><td>0.8 ± 0.04</td><td>0.8 ± 0.04</td><td>0.9 ± 0.04</td><td>0.95 ± 0.05</td><td>1.0 ± 0.0</td></tr><tr><td rowspan="4">HalfCheetah</td><td>umaze-diverse</td><td>0.92 ± 0.04</td><td>0.95 ± 0.05</td><td>0.9 ± 0.07</td><td>0.50 ± 0.25</td><td>0.9 ± 0.04</td><td>0.9 ± 0.07</td><td>0.9 ± 0.05</td><td>0.92 ± 0.04</td><td>1.0 ± 0.0</td></tr><tr><td>expert</td><td>1516.19 ± 25.85</td><td>1530.86 ± 21.28</td><td>1497.99 ± 48.14</td><td>1479.88 ± 22.09</td><td>1025.66 ± 23.74</td><td>1570.21 ± 27.80</td><td>1573.29 ± 46.12</td><td>1585.85±19.24</td><td>1538.32 ± 50.83</td></tr><tr><td>medium</td><td>655.96 ± 8.55</td><td>658.54 ± 2.02</td><td>667.08 ± 16.17</td><td>669.73 ± 15.74</td><td>597.75 ± 7.69</td><td>676.78 ± 3.14</td><td>664.15 ± 15.39</td><td>677.84 ± 13.6</td><td>679.03 ± 14.74</td></tr><tr><td>medium-expert</td><td>1432.58 ± 158.2</td><td>1572.42 ± 68.04</td><td>1619.32 ± 43.02</td><td>1103.51 ± 141.89</td><td>1097.76 ± 68.47</td><td>1541.82 ± 58.53</td><td>1566.91 ± 83.36</td><td>1636.29±27.28</td><td>1541.19 ± 73.06</td></tr><tr><td rowspan="4">Hopper</td><td>medium-replay</td><td>806.58 ± 175.57</td><td>1074.72 ± 11.14</td><td>1100.29 ± 15.2</td><td>946.68 ± 91.13</td><td>607.06 ± 110.68</td><td>749.42 ± 13.93</td><td>916.15 ± 67.71</td><td>862.53 ± 35.02</td><td>826.06 ± 96.8</td></tr><tr><td>expert</td><td>2129.4 ± 116.64</td><td>2234.25 ± 54.94</td><td>2239.86 ± 58.55</td><td>2072.71 ± 69.58</td><td>1211.95 ± 123.71</td><td>2183.71 ± 58.71</td><td>2218.42 ± 64.48</td><td>2260.68±22.36</td><td></td></tr><tr><td>medium</td><td>1144.2 ± 100.69</td><td>1241.6 ± 188.58</td><td>1252.95 ± 99.69</td><td>1175.3 ± 63.39</td><td>1101.02 ± 97.47</td><td>1299.97 ± 143.74</td><td>1289.51 ± 89.36</td><td>1383.51 ± 127.34</td><td>2230.41 ± 65.08</td></tr><tr><td>medium-expert</td><td>2056.43 ± 166.75</td><td>2270.67 ± 62.89</td><td>2259.2 ± 117.52</td><td>1549.4 ± 112.42</td><td>1128.34 ± 128.57</td><td>1776.79 ± 86.91</td><td>1914.85 ± 105.72</td><td>2026.48 ± 180.21</td><td>1231.19 ± 108.38</td></tr><tr><td rowspan="4">Kitchen</td><td>medium-replay</td><td>636.21 ± 91.45</td><td>564.7 ± 99.69</td><td>648.13 ± 172.74</td><td>378.57 ± 176.15</td><td>822.5 ± 86.30</td><td>1065.73 ± 90.73</td><td>1175.59 ± 95.76</td><td>1199.42 ± 192.37</td><td>1948.04 ± 342.99</td></tr><tr><td>complete</td><td>2.52 ± 0.18</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>770.51 ± 95.49</td></tr><tr><td>mixed</td><td>2.28 ± 0.27</td><td>2.53 ± 0.13</td><td>2.42 ± 0.38</td><td>2.7 ± 0.23</td><td>3.1 +0.16</td><td>2.5 ± 0.09</td><td>3.0 ± 0.02</td><td>2.58 ± 0.36</td><td>2.2 ± 0.14</td></tr><tr><td></td><td>2.65 ± 0.27</td><td>2.55 ± 0.21</td><td>2.15 ± 0.34</td><td>2.28 ± 0.13</td><td>2.5 ± 0.26</td><td>2.7 ± 0.18</td><td>2.5 ± 0.37</td><td>2.8 ± 0.2</td><td>2.65 ± 0.09</td></tr><tr><td rowspan="4">Maze2D</td><td>partial</td><td>106.72 ± 17.23</td><td>3.0 ± 0.07 103.7 ± 15.03</td><td>2.08 ± 0.51</td><td>2.42 ± 0.44</td><td>2.1 ± 0.25</td><td>3.0 ± 0.03</td><td>3.0 ± 0.02</td><td>3.02 ± 0.04</td><td>2.55 ± 0.43</td></tr><tr><td>large</td><td>40.07 ± 9.19</td><td></td><td>103.45 ± 19.85</td><td>34.1 ± 8.97</td><td>32.7 ± 16.05</td><td>69.9 ± 14.08</td><td>110.8 ± 17.74</td><td>79.65 ± 20.43</td><td>93.7 ± 71.77</td></tr><tr><td>medium</td><td></td><td>33.72 ± 13.3</td><td>154.0 ± 63.45</td><td>33.72 ± 7.87</td><td>37.8 ± 5.34</td><td>56.3 ± 5.35</td><td>111.9 ± 45.34</td><td>114.1 ± 75.94</td><td>145.23 ± 20.81</td></tr><tr><td>open</td><td>16.62 ± 0.48</td><td>17.17 ± 1.49</td><td>15.85 ± 3.61</td><td>19.08 ± 2.17</td><td>25.7 ± 1.64</td><td>30.2 ± 8.63</td><td>32.4 ± 4.56</td><td>32.33 ± 2.05</td><td>26.67 ± 0.76</td></tr><tr><td rowspan="4">Walker2D</td><td>umaze</td><td>58.68 ± 20.14 1863.25 ± 11.68</td><td>59.3 ± 20.61 1867.61 ± 9.43</td><td>61.2 ± 19.74</td><td>37.72 ± 11.7</td><td>58.8 ± 17.03</td><td>86.9 ± 10.47</td><td>186.1 ± 16.43</td><td>110.25 ± 51.04</td><td>31.13 ± 2.58</td></tr><tr><td>expert</td><td>1070.96 ± 61.03</td><td></td><td>1877.92 ± 5.12</td><td>1869.07 ± 8.54</td><td>1088.01 ± 14.72</td><td>1866.81 ± 4.72</td><td>1881.46 ± 9.46</td><td>1874.51 ± 10.86</td><td>1861.86 ± 7.2</td></tr><tr><td>medium</td><td></td><td>1081.54 ± 42.68</td><td>1175.83 ± 58.49</td><td>1106.63 ± 24.09</td><td>932.2 ± 40.81</td><td>1008.24 ± 54.91</td><td>1088.64 ± 43.09</td><td>1111.94 ± 32.08</td><td>1093.42 ± 36.83</td></tr><tr><td>medium-expert medium-replay</td><td>1310.26 ± 352.94 1129.27 ± 129.82</td><td>1883.23 ± 19.77 1335.64 ± 10.87</td><td>1880.87 ± 24.98 1377.9 ± 21.19</td><td>1153.5 ± 54.57 1197.66 ± 51.96</td><td>1055.61 ± 35.89 716.23 ± 78.38</td><td>1861.45 ± 23.23 1302.54 ± 12.94</td><td>1866.47 ± 18.67 1330.53 ± 34.71</td><td>1871.34 ± 4.88 1310.88 ± 32.54</td><td>1843.48 ± 11.49 1237.97 ± 63.21</td></tr></table></body></html>  

We train each model for 1 million epochs, using batch sizes of 16, and a learning rate of. $1e^{-4}$ . To mitigate the influence of outliers and the inherent seed-dependency of episodes, we adopt a validation strategy wherein, every one thousand epochs, we validate the model on 100 episodes, and compute the average accumulated rewards. Due to the seed dependency, we repeat each experiment across 4 different random seeds. The presented results are the average values. across the 4 seeds of the highest accumulated rewards seen throughout the 1 million epochs..  

We also varied the number of Mamba and Transformer layers, the embedding size inside the model, and the length of the token sequence. Results for DM, DM with RTG, and HDM are shown in Fig. 2, Fig. 4 and Fig. 5, respectively. The. values of the DM with RTG were obtained using a desired reward equal to the maximum reward present in the respective. demonstration data set. However, results were unclear, as there wasn't a best performing configuration for any of the.  

Table 2: Average and STD time for a single training iteration, and to perform inference of an episode, across the. different D4RL tasks, using a batch size of 16, of each of the 5 methods, configured with 6 layers, an embedding size of 128 and sequence length 20. Lowest values are highlighted in bold..   


<html><body><table><tr><td rowspan="2"></td><td colspan="5">TrainTime (s)</td><td colspan="5">Inference Time (s)</td></tr><tr><td>DT</td><td>HDT</td><td>DMwo/R</td><td>DMw/R</td><td>HDM</td><td>DT</td><td>HDT</td><td>DMwo/R</td><td>DMw/R</td><td>HDM</td></tr><tr><td>Ant</td><td>0.015±0.011</td><td>0.020±0.011</td><td>0.018±0.102</td><td>0.018±0.097</td><td>0.026±0.101</td><td>0.005±0.007</td><td>0.007±0.001</td><td>0.003±0.004</td><td>0.003±0.004</td><td>0.005±0.008</td></tr><tr><td>Antmaze</td><td>0.008±0.000</td><td>0.014±0.000</td><td>0.007±0.000</td><td>0.008±0.000</td><td>0.015±0.001</td><td>0.004±0.005</td><td>0.006±0.010</td><td>0.003±0.003</td><td>0.003±0.004</td><td>0.005±0.005</td></tr><tr><td>HalfCheetah</td><td>0.014±0.001</td><td>0.019±0.01</td><td>0.014三0.001</td><td>0.015±0.001</td><td>0.022±0.001</td><td>0.005±0.006</td><td>0.007±0.013</td><td>0.003±0.004</td><td>0.003±0.005</td><td>0.005±0.008</td></tr><tr><td>Hopper</td><td>0.012±0.001</td><td>0.017±0.001</td><td>0.012±0.001</td><td>0.012±0.001</td><td>0.020±0.001</td><td>0.005±0.006</td><td>0.008±0.012</td><td>0.003±0.004</td><td>0.003±0.004</td><td>0.005±0.007</td></tr><tr><td>Kitchen</td><td>0.008±0.000</td><td>0.014±0.000</td><td>0.009±0.000</td><td>0.009±0.000</td><td>0.017±0.001</td><td>0.006±0.003</td><td>9000干6000</td><td>0.005±0.003</td><td>0.005±0.003</td><td>0.006±0.004</td></tr><tr><td>Maze2d</td><td>0.008±0.000</td><td>0.014±0.000</td><td>0.008±0.000</td><td>0.008±0.000</td><td>0.015±0.001</td><td>0.003±0.003</td><td>90009000</td><td>0.003±0.003</td><td>0.003±0.003</td><td>0.004±0.004</td></tr><tr><td>Walker2d</td><td>0.013±0.001</td><td>0.020±0.001</td><td>0.014±0.001</td><td>0.014±0.001</td><td>0.022±0.001</td><td>0.005±0.007</td><td>0.006±0.012</td><td>0.003±0.004</td><td>0.003±0.005</td><td>0.005±0.008</td></tr></table></body></html>  

models. Different configurations resulted in better or worse performance for the models on the different tasks and data sets. To fairly compare the models using the same architecture, we chose the architecture that better represented the average results across the set of architectures. This architecture is composed of 6 layers (Transformer or Mamba), an embedding size of 128 and a sequence length of 20. For the execution of DT and DM using the sequence of RTG, we initially identify the maximum accumulated returns attained by a trajectory in the demonstration data set. Subsequently, during validation, we set the desired returns to this maximum value, half of it, and a larger value--specifically, 10k. The results are presented in Table 1.  

One of the driving motivations behind the development of the HDT was to alleviate the necessity of manually specifying the desired RTG, a notable challenge encountered in the evaluation and deployment of DTs. To determine whether the DM inherits this drawback from DT, we evaluate whether it also relies on the RTG sequence to guide the model, or if the sequence can be simply removed from the model's input. Table 1 presents the accumulated returns achieved by the DM model without the desired returns sequence, compared with a variant of the DM model with this additional sequence.  

The results depicted in the table highlight that the DM does not require the sequence of RTG for effective performance.. Additionally, results also show that the DM without the sequence of RTG does seem to reach a slightly better performance than with the sequence. This is also true across different architectures as shown in Fig. 4. Notably, in the DT, eliminating this sequence impedes the DT's ability to learn the task entirely as shown in [4]. This indicates that the evolutionary parameter of the Mamba architecture successfully replaces the need for RTG. Since DM without the. sequence of rewards achieves higher performance without requiring additional user interaction, we can conclude that DM should be used without the reward sequence.  

Lastly, we compare the new proposed Mamba methods with their Transformer predecessors. According to the results in Table 1, the DM without rewards outperforms the DT in 15 out of the 27 settings. Additionally, the HDM outperforms the HDT in 22 out of the 27 settings. The superiority of the Mamba models exists even while comparing to the DT with the ideal desired reward for each task. When comparing the DM to the DT using a fixed desired reward of 10k, the DM. outperforms the DT in 17 out of the 27 settings. These results show that the proposed Mamba methods improve upon the Transformer predecessors in the D4RL benchmark. Overall, the DM without rewards is the best performing model of the set. Although, the HDM and the DT are still very competitive, it is worth noting that HDM requires two models and pre-processing the data set, while DT requires user interaction and task knowledge. Moreover, unlike the other. models, DM can be applied to tasks without a reward function..  

# 5.1 Time Comparison  

Mamba models have outpaced Transformers in terms of speed in other applications. Also, HDT and HDM require the training of two models, and the extra computational cost may not be worth the performance benefits. Because of this,. we compare the time required to perform a training iteration and the inference step using the different methods across. the 7 task environments available in the D4RL benchmark. We use a batch size of 16, an embedding size of 128, a sequence length of 20, and 6 layers per model. Specifically, for the HDT and the HDM, we employ 6 layers for both the high-level and low-level models, and we measure the time to train both models. For training, we measure the time it. takes for a gradient calculation and update. For inference, we measure the time it takes to build the sequences, obtain an action from the model and perform the transition. To ensure statistical robustness, we repeat both these steps 1000 times for each model, presenting the average and standard deviation time to perform a training iteration, and an inference step in Table 2. Results show that during training there's not much difference between the Mamba methods and their Transformer predecessors. As expected the HDT and the HDM take close to double the time to train due to having double the models than the DT and the DM, respectively. Adding rewards to the DM does not increase the training time significantly. At inference time however, the Mamba methods are faster than the Transformer methods. In addition to the increase in performance, this computational boost further shows the benefits of our methods compared to the baselines.

# References  

[1] Sutton, R. & Barto, A. Reinforcement learning: An introduction. (MIT press,2018) [2] Fu, J., Kumar, A., Nachum, O., Tucker, G. & Levine, S. D4rl: Datasets for deep data-driven reinforcement learning. ArXiv Preprint arXiv:2004.07219. (2020) [3] Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A. & Mordatch, I. Decision transformer: Reinforcement learning via sequence modeling. Advances In Neural Information Processing Systems. 34 pp. 15084-15097 (2021) [4] Correia, A. & Alexandre, L. Hierarchical decision transformer. 2023 IEEE/RSJ International Conference On Intelligent Robots And Systems (IROS). pp. 1661-1666 (2023)   
[5] Gu, A. & Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. ArXiv Preprint arXiv:2312.00752. (2023)   
[6] Gu, A., Goel, K. & Re, C. Efficiently modeling long sequences with structured state spaces. ArXiv Preprint arXiv:2111.00396. (2021) [7] Bhirangi, R., Wang, C., Pattabiraman, V., Majidi, C., Gupta, A., Hellebrekers, T. & Pinto, L. Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling. ArXiv Preprint arXiv:2402.10211. (2024)   
[8] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. & Others Language models are unsupervised multitask learners. OpenAI Blog. 1, 9 (2019) [9] Hansen, N., Su, H. & Wang, X. Stabilizing deep q-learning with convnets and vision transformers under data augmentation. Advances In Neural Information Processing Systems. 34 pp. 3680-3693 (2021)   
[10] Argall, B., Chernova, S., Veloso, M. & Browning, B. A survey of robot learning from demonstration. Robotics And Autonomous Systems. 57, 469-483 (2009)   
[11] Ross, S., Gordon, G. & Bagnell, D. A reduction of imitation learning and structured prediction to no-regret online learning. Proceedings Of The Fourteenth International Conference On Artificial Intelligence And Statistics. pp. 627-635 (2011)   
[12] Mandlekar, A., Ramos, F., Boots, B., Savarese, S., Fei-Fei, L., Garg, A. & Fox, D. Iris: Implicit reinforcement. without interaction at scale for learning control from offline robot manipulation data. 2020 IEEE International Conference On Robotics And Automation (ICRA). pp. 4414-4420 (2020)   
[13] Krishnan, S., Garg, A., Liaw, R., Thananjeyan, B., Miller, L., Pokorny, F. & Goldberg, K. SwIRL: A Sequential Windowed Inverse Reinforcement Learning Algorithm for Robot Tasks With Delayed Rewards. Algorithmic. Foundations Of Robotics XII: Proceedings Of The Twelfth Workshop On The Algorithmic Foundations Of Robotics.. pp. 672-687 (2020)   
[14] Chane-Sane, E., Schmid, C. & Laptev, I. Goal-conditioned reinforcement learning with imagined subgoals. International Conference On Machine Learning. pp. 1430-1440 (2021)   
[15] Janner, M., Li, Q. & Levine, S. Offline reinforcement learning as one big sequence modeling problem. Advances In Neural Information Processing Systems. 34 pp. 1273-1286 (2021)   
[16] Zheng, Q., Zhang, A. & Grover, A. Online decision transformer. International Conference On Machine Learning. pp. 27042-27059 (2022)   
[17] Villaflor, A., Huang, Z., Pande, S., Dolan, J. & Schneider, J. Addressing optimism bias in sequence modeling for reinforcement learning. International Conference On Machine Learning. pp. 22270-22283 (2022)   
[18] Yu, T., Kumar, A., Chebotar, Y., Hausman, K., Levine, S. & Finn, C. Conservative data sharing for multi-task offline reinforcement learning. Advances In Neural Information Processing Systems. 34 pp. 11501-11516 (2021)   
[19] Reid, M., Yamada, Y. & Gu, S. Can wikipedia help offline reinforcement learning?. ArXiv Preprint. arXiv:2201.12122. (2022)   
20] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A., Kaiser, L. & Polosukhin, I. Attention i all you need. Advances In Neural Information Processing Systems. 30 (2017)   
[21] Hsu, H., Bozkurt, A., Dong, J., Gao, Q., Tarokh, V. & Pajic, M. Steering Decision Transformers via Temporal. Difference Learning.   
[22] Kumar, A., Zhou, A., Tucker, G. & Levine, S. Conservative q-learning for offline reinforcement learning. Advances In Neural Information Processing Systems. 33 pp. 1179-1191 (2020)  


###输出

#Reference List:
	1.	“DT”： “Decision Transformer: Reinforcement Learning via Sequence Modeling” ，“Arxiv地址”： https://arxiv.org/abs/2106.01345
	2.	“HDT”： “Hierarchical Decision Transformer” ，“Arxiv地址”： 无
	3.	“DM/HDM”： “Mamba: Linear-time sequence modeling with selective state spaces” ，“Arxiv地址”： https://arxiv.org/abs/2312.00752

##技巧

你可以重点关注Table及其html格式中出现的模型、实验部分反复提到的模型

""")

