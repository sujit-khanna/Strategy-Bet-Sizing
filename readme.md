# Betsizing algorithm using machine learning classifier for intraday VIX trading strategies
 - This repo contains implementation of a bet-sizing algorithm for a portfolio of intraday trading strategies on long VIX ETF (VXX)
 - The presentation.pdf contains a detailed discussion of the methodology and results
 - The size of the bet is calculated using the classifier of ML model and kelly criteria
 - feature_modelling module contains all the code for computing bet-sizing features
 - Raw strategy backtests, Price-Volume data is saved in data folder, please unzip the folder using command "unzip data.zip" for linux, or use winzip for windows
 - Betsizing_features.ipynb notebook contains the code for creating bet-sizing features, this ntebook also saves the files for computing the optimal leverage of the strategies
 - shap_analysis_bet_size_labeling.ipynb contains analysis of features used in this project
 - betsized_strategies.ipynb contains the actual implementation of the bet_sizing algorithm for each strategy and analysis the performance at the portfolio level

## Running the project
 - The project requires python3.7; so please ensure you're working on this version of python
 - Install the required dependencies using the command: pip3.7 install -r requirements.txt
 - Please unzip data.zip folder using command "unzip data.zip" for linux, or use winzip for windows
 - To reproduce the results, first run the betsizing_features.ipynb notebook to generate the feature files, then run betsizing_strategies.ipynb to generate bet sized results
 - Individual features for a given strategy can be analyzed in the shap_analysis_bet_size_labeling.ipynb notebook
 