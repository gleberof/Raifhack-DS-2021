# Raifhack-DS-2021

[RaifHack 2021](https://raifhack.ru/) competition in commersial real estate price prediction winning solution from the "Звездочка" team.

The main idea of the solution is using a cascade of two models trained with out-of-fold validation. The first model is trained to predict the price_0, while the second model corrects the first model outputs to estimate price_1. Additionally, each of the models is actually a blend of a LightGBM and a [TabNet](https://arxiv.org/abs/1908.07442). 

The TabNet (attention-based) architecture allows for prediction explaination as well as extracting feature importances.
