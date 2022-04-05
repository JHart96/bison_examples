data {
  int N; // Number of edges
  vector[N] y_mu; // Means of Gaussian approximations of centrality
  vector[N] y_sigma; // Standard deviations of Gaussian approximations of centrality
  int dyadtypes[N]; // Dyad types of edges
  int node_ids[N]; // Node IDs
}

parameters {
  real intercept;
  vector[3] beta_dyadtype;
  real<lower=0> sigma;
}

model {
  y_mu ~ normal(intercept + beta_dyadtype[dyadtypes], sqrt(square(y_sigma) + square(sigma))); // Should this be sqrt(y_sigma^2 + sigma^2)?

  intercept ~ normal(0, 1);
  beta_dyadtype ~ normal(0, 1);
  sigma ~ normal(0, 1);
}
