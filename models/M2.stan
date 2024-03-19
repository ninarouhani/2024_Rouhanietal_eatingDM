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
    real                      pop_alpha_pos;
    real                      pop_alpha_neg;
    vector<lower=0>[3]        sigma; 
    
    vector[nSubj]             prior_beta;                
    vector[nSubj]             prior_alpha_pos;    
    vector[nSubj]             prior_alpha_neg;  
    
}
transformed parameters {
 
    vector<lower=0, upper=1>[nSubj]  alphas_pos;
    vector<lower=0, upper=1>[nSubj]  alphas_neg;
    vector<lower=0, upper=4>[nSubj]  betas;
 
    for (s in 1:nSubj) {
      betas[s] = Phi_approx(pop_beta + sigma[1] * prior_beta[s])*4;
      alphas_pos[s] = Phi_approx(pop_alpha_pos + sigma[2] * prior_alpha_pos[s]);
      alphas_neg[s] = Phi_approx(pop_alpha_neg + sigma[3] * prior_alpha_neg[s]);
    }
    
}
model {
    real delta;
    real alpha;
 
    // initialize Q-values
    vector[2] Q;
    
    pop_beta ~ normal(0,1);
    pop_alpha_pos ~ normal(0,1);
    pop_alpha_neg ~ normal(0,1);
    sigma ~ cauchy(0,5);
    
    prior_beta ~ normal(0,1);
    prior_alpha_pos ~ normal(0,1);
    prior_alpha_neg ~ normal(0,1);

    for (n in 1:maxTrials) {
      
      if (trialNum[n]==1){
        Q = rep_vector(0, 2);
      }
      
      choiceInd[n] ~ categorical_logit(betas[subjInd[n]]*Q);
            
      // print(n,",",Q,choiceInd[n],",",dEV[n]);
            
      // compute prediction error
      delta = reward[n] - Q[choiceInd[n]];
        
      if (delta<0){
        alpha = alphas_neg[subjInd[n]];
      }else{
        alpha = alphas_pos[subjInd[n]];
      }
        
      // Update state-action values
      Q[choiceInd[n]] += alpha * delta;
                
    }

}
generated quantities {
    real<lower=0, upper=1> mu_a_pos;
    real<lower=0, upper=1> mu_a_neg;
    real<lower=0, upper=4> mu_beta;
    vector[maxTrials] log_lik;
    //vector[maxTrials] y_pred;
    real delta;
    real alpha;
 
    // initialize Q-values
    vector[2] Q;

    mu_a_pos  = Phi_approx(pop_alpha_pos);
    mu_a_neg  = Phi_approx(pop_alpha_neg);
    mu_beta = Phi_approx(pop_beta)*4;

    for (n in 1:maxTrials) {
      
      if (trialNum[n]==1){
        Q = rep_vector(0, 2);
      }
      
      // compute log likelihood of current trial
      log_lik[n] = categorical_logit_lpmf(choiceInd[n] | betas[subjInd[n]]*Q);
  
      // generate posterior prediction for current trial
      //y_pred[n] = categorical_rng(softmax(betas[subjInd[n]]*Q));  
    
      // compute prediction error
      delta = reward[n] - Q[choiceInd[n]];
        
      if (delta<0){
        alpha = alphas_neg[subjInd[n]];
      }else{
        alpha = alphas_pos[subjInd[n]];
      }
        
      // Update state-action values
      Q[choiceInd[n]] += alpha * delta;
                
    }

}
