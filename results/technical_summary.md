# Technical Summary

Research question: Do technical indicators help RL trading under transaction costs?

The study compares identical DQN agents across controlled feature groups and includes Q-Learning plus non-RL baselines.


## Mean DQN test metrics

                   cumulative_return    sharpe  max_drawdown  turnover
feature_group                                                         
F1_returns_only                 -1.0  1.088844          -1.0  0.075610
F2_returns_volume               -1.0  1.076649          -1.0  0.073984
F3_momentum                     -1.0  1.086001          -1.0  0.044715
F4_trend                        -1.0  1.070475          -1.0  0.052846
F5_volatility                   -1.0  1.092384          -1.0  0.073984
F6_all_indicators               -1.0  1.034938          -1.0  0.088618



## Mean indicator effects versus returns-only DQN

                   delta_cumulative_return  delta_sharpe  drawdown_improvement  delta_turnover
feature_group                                                                                 
F2_returns_volume             4.847778e-11     -0.012195          3.752554e-14       -0.001626
F3_momentum                   4.021470e-08     -0.002843         -2.209344e-14       -0.030894
F4_trend                      1.335763e-08     -0.018369         -3.460565e-13       -0.022764
F5_volatility                 2.339126e-07      0.003540         -5.706768e-12       -0.001626
F6_all_indicators            -2.558282e-09     -0.053906          4.485301e-14        0.013008



## Statistical tests

    test              status
wilcoxon scipy_not_installed



Interpretation rule: indicators help only if they improve out-of-sample Sharpe/return without unacceptable drawdown or turnover increase.
