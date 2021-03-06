# Class Imbalance Demo on load_breast_cancer

**Objective**: show the impact of the class imbalance on the features importance  <br>

**Comment**: Experiments "Without Class Imbalance Correction" and "Slight Class Imbalance" have similar features importance, albeit not identical, and the roc auc score for both is close. Experiment "Extreme Class Imbalance" leads to significant different features importance with a model that does not outperform a random guess (roc auc score = 0.5)

## Without Class Imbalance Correction

<p align="middle">
  <img src="https://github.com/dagrate/classimbalance/blob/master/plots/without.png" width="800"/>
</p>

## With Small Class Imbalance

<p align="middle">
  <img src="https://github.com/dagrate/classimbalance/blob/master/plots/slight.png" width="800"/>
</p>

## With Extreme Class Imbalance

<p align="middle">
  <img src="https://github.com/dagrate/classimbalance/blob/master/plots/extreme.png" width="800"/>
</p>
