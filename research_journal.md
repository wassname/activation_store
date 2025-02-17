# 2025-02-17 17:39:09

My supressed activation experiment on 3b and 7b models worked

TODO
- [ ] Graph
- [ ] Try other formulation of supression
  - [ ] Mark each neuron?
  - [ ] Graph by token?
  - [ ] check the projections are communtitive and reversible!?
  - [ ] stats on supressed neurons (histogram)
  - [ ] what about multi layer decoder, other datasets, more samples


3B model
LLM score: 0.53 roc auc, n=116

	name	score
3	hs_sup last	0.639881
17	supressed_mask none	0.631845
15	supressed_mask last	0.626786
1	hs_sup max	0.619643
11	hidden_states none	0.617559
0	hs_sup mean	0.615476
2	hs_sup sum	0.615476
8	hidden_states sum	0.608333
6	hidden_states mean	0.608333
5	hs_sup none	0.602679
16	supressed_mask first	0.601786
10	hidden_states first	0.594345
4	hs_sup first	0.572024
9	hidden_states last	0.557143
14	supressed_mask sum	0.554762
12	supressed_mask mean	0.554762
7	hidden_states max	0.541964
13	supressed_mask max	0.496726



0.5b model
LLM score: 0.54 roc auc, n=116

	name	score
33	hs_sup none last	0.769643
23	hs_sup last none	0.769643
100	hidden_states none mean	0.755059
93	hidden_states last none	0.755059
87	hidden_states sum none	0.755059
105	hidden_states none none	0.755059
103	hidden_states none last	0.755059
102	hidden_states none sum	0.755059
99	hidden_states first none	0.754762
101	hidden_states none max	0.754762
75	hidden_states mean none	0.754464
104	hidden_states none first	0.754464
81	hidden_states max none	0.754167
11	hs_sup max none	0.753274
