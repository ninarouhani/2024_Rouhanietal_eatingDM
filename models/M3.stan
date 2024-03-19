data {
	int nSubj;
	int maxTrials;

  int subjInd[maxTrials];
  int choiceInd[maxTrials];
	int reward[maxTrials];
	int trialNum[maxTrials];

}
parameters {
    real                      pop_beta;                 
    real                      pop_alpha;
    real<lower=0, upper=1>    pres;
    vector<lower=0>[3]        sigma; 
    
    vector[nSubj]             prior_beta;                
    vector[nSubj]             prior_alpha;    
    vector[nSubj]             prior_pres;  
    
}
transformed parameters {
 
    vector<lower=0, upper=1>[nSubj]  alphas;
    vector<lower=0, upper=4>[nSubj]  betas;
    vector<lower=0, upper=1>[nSubj]  pres_p;
 
    for (s in 1:nSubj) {
      betas[s] = Phi_approx(pop_beta + sigma[1] * prior_beta[s])*4;
      alphas[s] = Phi_approx(pop_alpha + sigma[2] * prior_alpha[s]);
      pres_p[s] = pres + sigma[3] * prior_pres[s];
    }
    
}
model {
    real delta;
 
    // initialize Q-values
    vector[2] Q;
    
    pop_beta ~ normal(0,1);
    pop_alpha ~ normal(0,1);
    pres ~ normal(0,1);
    sigma ~ cauchy(0,5);
    
    prior_beta ~ normal(0,1);
    prior_alpha ~ normal(0,1);
    prior_pres ~ normal(0,1);
  
    for (n in 1:maxTrials) {
      
      if (trialNum[n]==1){
        Q = rep_vector(0, 2);
      }else{
          Q[choiceInd[n-1]] = Q[choiceInd[n-1]] + pres_p[subjInd[n]];
      }
      
      choiceInd[n] ~ categorical_logit(betas[subjInd[n]]*Q);
            
      // print(n,",",Q,choiceInd[n],",",dEV[n]);
            
      // compute prediction error
      delta = reward[n] - Q[choiceInd[n]];
        
      // Update state-action values
      Q[choiceInd[n]] += alphas[subjInd[n]] * delta;
                
    }

}
generated quantities {
    real<lower=0, upper=1> mu_a;
    real<lower=0, upper=4> mu_beta;
    vector[maxTrials] log_lik;
    //vector[maxTrials] y_pred;
    real delta;
 
    // initialize Q-values
    vector[2] Q;

    mu_a    = Phi_approx(pop_alpha);
    mu_beta = Phi_approx(pop_beta)*4;

    for (n in 1:maxTrials) {
      
      if (trialNum[n]==1){
        Q = rep_vector(0, 2);
      }else{
        Q[choiceInd[n-1]] = Q[choiceInd[n-1]] + pres_p[subjInd[n]];
      }
      
      // compute log likelihood of current trial
      log_lik[n] = categorical_logit_lpmf(choiceInd[n] | betas[subjInd[n]]*Q);
  
      // generate posterior prediction for current trial
      //y_pred[n] = categorical_rng(softmax(betas[subjInd[n]]*Q));  
    
      // compute prediction error
      delta = reward[n] - Q[choiceInd[n]];
        
      // Update state-action values
      Q[choiceInd[n]] += alphas[subjInd[n]] * delta;
                
    }

}
