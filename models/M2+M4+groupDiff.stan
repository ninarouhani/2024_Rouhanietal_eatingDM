data {
	int nSubj;
	int maxTrials;

  int subjInd[maxTrials];
  int choiceInd[maxTrials];
	int reward[maxTrials];
	int trialNum[maxTrials];
	int group[maxTrials];
	int blockInd[maxTrials];

}
parameters {
    real                      pop_beta;                 
    real                      pop_alpha_pos;
    real                      pop_alpha_neg;
    real<lower=-1, upper=1>   diff_lc_b1;
    real<lower=-1, upper=1>   diff_hc_b1;
    real                      qdiff_rel_b1;
    real<lower=-1, upper=1>   diff_lc_b2;
    real<lower=-1, upper=1>   diff_hc_b2;
    real                      qdiff_rel_b2;
    real                      pop_q;
    vector<lower=0>[4]        sigma; 
    
    vector[nSubj]             prior_beta;                
    vector[nSubj]             prior_alpha_pos;    
    vector[nSubj]             prior_alpha_neg;  
    vector[nSubj]             prior_q;  
    
}
transformed parameters {
 
    vector<lower=0, upper=1>[nSubj]  alphas_pos;
    vector<lower=0, upper=1>[nSubj]  alphas_neg;
    vector<lower=0, upper=4>[nSubj]  betas;
    vector[nSubj]  qs;
 
    for (s in 1:nSubj) {
      betas[s] = Phi_approx(pop_beta + sigma[1] * prior_beta[s])*4;
      alphas_pos[s] = Phi_approx(pop_alpha_pos + sigma[2] * prior_alpha_pos[s]);
      alphas_neg[s] = Phi_approx(pop_alpha_neg + sigma[3] * prior_alpha_neg[s]);
      qs[s] = pop_q + sigma[4] * prior_q[s];
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
    pop_q ~ normal(0,1);
    
    diff_lc_b1 ~ normal(0,1);
    diff_hc_b1 ~ normal(0,1);
    qdiff_rel_b1 ~ normal(0,1);
    
    diff_lc_b2 ~ normal(0,1);
    diff_hc_b2 ~ normal(0,1);
    qdiff_rel_b2 ~ normal(0,1);
    
    sigma ~ cauchy(0,5);
    
    prior_beta ~ normal(0,1);
    prior_alpha_pos ~ normal(0,1);
    prior_alpha_neg ~ normal(0,1);
    prior_q ~ normal(0,1);  

    for (n in 1:maxTrials) {
      
      if (trialNum[n]==1){
        if (group[n]==0&&blockInd[n]==2){
          Q[1] = qs[subjInd[n]]+qdiff_rel_b2;
          Q[2] = qs[subjInd[n]];
        }else if(group[n]==0&&blockInd[n]==1){
          Q[1] = qs[subjInd[n]]+qdiff_rel_b1;
          Q[2] = qs[subjInd[n]];
        }else{
          Q[1] = qs[subjInd[n]];
          Q[2] = qs[subjInd[n]];
        }
      }
      
      choiceInd[n] ~ categorical_logit(betas[subjInd[n]]*Q);
            
      //print(n,",",Q,choiceInd[n]);
            
      // compute prediction error
      delta = reward[n] - Q[choiceInd[n]];
        
      if (delta<0){
        alpha = alphas_neg[subjInd[n]];
      }else{
        if (group[n]==0&&choiceInd[n]==1&&blockInd[n]==2){
          alpha = alphas_pos[subjInd[n]] + diff_lc_b2;
        }else if(group[n]==0&&choiceInd[n]==2&&blockInd[n]==2){
          alpha = alphas_pos[subjInd[n]] + diff_hc_b2;
        }else if(group[n]==0&&choiceInd[n]==2&&blockInd[n]==1){
          alpha = alphas_pos[subjInd[n]] + diff_hc_b1;
        }else if(group[n]==0&&choiceInd[n]==1&&blockInd[n]==1){
          alpha = alphas_pos[subjInd[n]] + diff_lc_b1;
        }else{
          alpha = alphas_pos[subjInd[n]];
        }
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
    vector[maxTrials] y_pred;
    real delta;
    real alpha;
 
    // initialize Q-values
    vector[2] Q;

    mu_a_pos  = Phi_approx(pop_alpha_pos);
    mu_a_neg  = Phi_approx(pop_alpha_neg);
    mu_beta = Phi_approx(pop_beta)*4;

    for (n in 1:maxTrials) {
      
      if (trialNum[n]==1){
        if (group[n]==0&&blockInd[n]==2){
          Q[1] = qs[subjInd[n]]+qdiff_rel_b2;
          Q[2] = qs[subjInd[n]];
        }else if(group[n]==0&&blockInd[n]==1){
          Q[1] = qs[subjInd[n]]+qdiff_rel_b1;
          Q[2] = qs[subjInd[n]];
        }else{
          Q[1] = qs[subjInd[n]];
          Q[2] = qs[subjInd[n]];
        }
      }
      
      // compute log likelihood of current trial
      log_lik[n] = categorical_logit_lpmf(choiceInd[n] | betas[subjInd[n]]*Q);
  
      // generate posterior prediction for current trial
      y_pred[n] = categorical_rng(softmax(betas[subjInd[n]]*Q));  
    
      // compute prediction error
      delta = reward[n] - Q[choiceInd[n]];
        
      if (delta<0){
        alpha = alphas_neg[subjInd[n]];
      }else{
        if (group[n]==0&&choiceInd[n]==1&&blockInd[n]==2){
          alpha = alphas_pos[subjInd[n]] + diff_lc_b2;
        }else if(group[n]==0&&choiceInd[n]==2&&blockInd[n]==2){
          alpha = alphas_pos[subjInd[n]] + diff_hc_b2;
        }else if(group[n]==0&&choiceInd[n]==2&&blockInd[n]==1){
          alpha = alphas_pos[subjInd[n]] + diff_hc_b1;
        }else if(group[n]==0&&choiceInd[n]==1&&blockInd[n]==1){
          alpha = alphas_pos[subjInd[n]] + diff_lc_b1;
        }else{
          alpha = alphas_pos[subjInd[n]];
        }
      }
        
      // Update state-action values
      Q[choiceInd[n]] += alpha * delta;
                
    }

}
