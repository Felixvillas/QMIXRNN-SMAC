# QMIXRNN
Referring to pymarl, qmix is implemented clearly with RNN to cope with SMAC environment.
This clear implementation can help you figure out how does QMIX work  

## Run
Note: --optimizer=0/1 means that optimizer `Adam` and `RMSprop` is good on this scenario, please just select one of both when running
`python main.py --map-name=3s5z --optimizer=0/1`
`python main.py --map-name=1c3s5z --optimizer=0/1`
`python main.py --map-name=2s3z --optimizer=0/1`
`python main.py --map-name=8m --optimizer=0/1`
`python main.py --map-name=2s_vs_1sc --optimizer=0`
`python main.py --map-name=3m --optimizer=0`
`python main.py --map-name=10m_vs_11m --optimizer=0`

## TODO
Now this code can do very good on part of easy scenarios like 1c3s5z, 2s3z, 3s5z and 8m, 
and relative good on easy scenarios like 2s_vs_1sc and 3m,
but not good on easy scenarios 10m_vs_11m.

I'm trying to approach the result of pymarl. At the same time, I'm also trying to achieve some tricks on this code like multi step TD target and so on. 

## Reference
@inproceedings{rashid2018qmix,
  title={Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning},
  author={Rashid, Tabish and Samvelyan, Mikayel and Schroeder, Christian and Farquhar, Gregory and Foerster, Jakob and Whiteson, Shimon},
  booktitle={International conference on machine learning},
  pages={4295--4304},
  year={2018},
  organization={PMLR}
}

@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}

@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}