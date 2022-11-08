data {
  int<lower=0> N; // Number of data points
  int<lower=0> M; // Number of dyads
  int<lower=0> dyad_ids[N]; // Dyad ID corresponding to each data point
  int<lower=0, upper=1> event[N]; // Outcome corresponding to each data point (presence/absence)
}

parameters {
  vector<lower=0, upper=1>[M] edge_weight; // Edge weights for each dyad.
  real<lower=0, upper=1> prob_detect; // Detection probability.
}

model {
  // Main model
  event ~ bernoulli(prob_detect^2 * edge_weight[dyad_ids]);

  // Priors
  edge_weight ~ beta(1, 1);
  prob_detect ~ beta(1, 1);
}

generated quantities {
  int event_pred[N] = bernoulli_rng(prob_detect^2 * edge_weight[dyad_ids]);
}
