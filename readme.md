# Class Imbalance Demo on load_breast_cancer

Objective: show the impact on the feature importance of the class imbalance <br>
Comment: Without Class Imbalance Correction and Slight Class Imbalance have similar feature importance albeit not identical and the roc auc score for both is close. Extreme class imbalance leads to significant different feature importance with a model that does not outperform a random guess (roc auc score = 0.5)

## Without Class Imbalance Correction

<p align="middle">
  <img src="https://github.com/dagrate/classimbalance/blob/master/plots/without.png" width="800"/>
</p>

## With Extreme Class Imbalance

<p align="middle">
  <img src="https://github.com/dagrate/classimbalance/blob/master/plots/extreme.png" width="800"/>
</p>

## With Small Class Imbalance

<p align="middle">
  <img src="https://github.com/dagrate/classimbalance/blob/master/plots/slight.png" width="800"/>
</p>
