# Technical Summary

Research question: Do technical indicators help RL trading under transaction costs?

The study compares identical DQN agents across controlled feature groups and includes Q-Learning plus non-RL baselines.


## Mean DQN test metrics

                   cumulative_return    sharpe  max_drawdown  turnover
feature_group                                                         
F1_returns_only           113.862388  0.881313     -0.999984  0.454472
F2_returns_volume          34.746397  0.954485     -0.999999  0.245528
F3_momentum                -0.830413  1.449812     -0.999997  0.172900
F4_trend               184319.147931  0.863744     -1.000000  0.416531
F5_volatility              -0.987244  1.350234     -1.000000  0.251762
F6_all_indicators          11.291989  1.642321     -0.999935  0.240379



## Mean indicator effects versus returns-only DQN

                   delta_cumulative_return  delta_sharpe  drawdown_improvement  delta_turnover
feature_group                                                                                 
F2_returns_volume               -79.115991      0.073172              0.000015       -0.208943
F3_momentum                    -114.692801      0.568498              0.000012       -0.281572
F4_trend                     184205.285543     -0.017570              0.000016       -0.037940
F5_volatility                  -114.849633      0.468921              0.000016       -0.202710
F6_all_indicators              -102.570400      0.761008             -0.000049       -0.214092



## Statistical tests

    test              status
wilcoxon scipy_not_installed



Interpretation rule: indicators help only if they improve out-of-sample Sharpe/return without unacceptable drawdown or turnover increase.
