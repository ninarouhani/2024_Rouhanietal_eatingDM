library(rstan)
library(loo)
library(dplyr)
setwd('../data/')
data = read.csv('data_choice.csv')
data = subset(data, block=='food')

## PREPARE DATA FOR STAN ##
subs = unique(data$participant)
nSubj = length(subs)

# group trial-length indicator
group = data$group2 # 1=LED/0=HED
trialNum = data$trialNumber
reward = data$reward
maxTrials = length(reward)
subjInd = data$participant
choiceInd = data$choiceInd # 1=low-cal choice, 2=high-cal choice
blockInd = data$block

standata = list(nSubj=nSubj, subjInd=subjInd, group=group, choiceInd=choiceInd, blockInd = blockInd, maxTrials=maxTrials, reward=reward, trialNum=trialNum)

## RUN ##
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

fit<- stan(file = 'M0.stan', data = standata, iter = 5000, control=list(adapt_delta = 0.8,max_treedepth=30), chains = 4, cores=4)

# WAIC calculation 
log_lik<-extract_log_lik(fit)
waic(log_lik)
