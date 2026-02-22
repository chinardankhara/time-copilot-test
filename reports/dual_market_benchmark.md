# Dual-Market Time Copilot Benchmark Report

## Champion Summary
| market | task | model | primary_metric | primary_value |
| --- | --- | --- | --- | --- |
| PJM | forecast | gbdt_reg | smape | 12.01095817237453 |
| PJM | rally | gbdt_cls | pr_auc | 0.8700866082225014 |
| NP | forecast | dnn_reg | smape | 5.538151790678604 |
| NP | rally | dnn_cls | pr_auc | 0.7890682430785442 |

## Market Details
### NP

Forecast benchmark:
| model | mae | rmse | smape |
| --- | --- | --- | --- |
| dnn_reg | 2.111234947739207 | 3.890769984240037 | 5.538151790678604 |
| gbdt_reg | 2.228493910502859 | 4.082862034998294 | 5.615490076041754 |
| lear | 2.217711322438352 | 4.211488950449745 | 5.862260901037497 |
| naive | 2.5371215311004787 | 4.958049119539895 | 6.535583645917815 |

Rally benchmark:
| model | pr_auc | roc_auc | f1 | brier |
| --- | --- | --- | --- | --- |
| dnn_cls | 0.7890682430785442 | 0.8839547937343946 | 0.7109919571045576 | 0.1407681585680448 |
| logreg | 0.7870298882777027 | 0.8891451529494011 | 0.7038845726970033 | 0.1784520100563235 |
| gbdt_cls | 0.6958636808069092 | 0.8168566049496266 | 0.6155609321278492 | 0.1799041002549391 |
| naive | 0.3251674641148325 | 0.5 | 0.4907567879838244 | 0.2218674546264297 |

### PJM

Forecast benchmark:
| model | mae | rmse | smape |
| --- | --- | --- | --- |
| gbdt_reg | 3.159369745407522 | 5.137598814929919 | 12.01095817237453 |
| dnn_reg | 3.4708247308068834 | 5.370952599596339 | 13.20035513495944 |
| lear | 4.192356382595326 | 6.146877432276291 | 16.125403344490906 |
| naive | 4.950168123540671 | 7.274897928712537 | 18.361299626605756 |

Rally benchmark:
| model | pr_auc | roc_auc | f1 | brier |
| --- | --- | --- | --- | --- |
| gbdt_cls | 0.8700866082225014 | 0.9142064402874608 | 0.7656624227908736 | 0.1155806914504539 |
| dnn_cls | 0.8531901950695358 | 0.8945085600495243 | 0.7578681369170687 | 0.13726428080074 |
| logreg | 0.8154865156799441 | 0.8921164327739378 | 0.76003276003276 | 0.1367632024829465 |
| naive | 0.3766507177033493 | 0.5 | 0.5471986653691089 | 0.2351539324019896 |

## Business Interpretation
- The same forecasting stack can map to freight and refinery decision workflows by replacing targets and desk-specific features.
- Champion selection is automated per market/task, enabling repeatable model governance.
