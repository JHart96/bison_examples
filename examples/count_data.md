Count Data Example
================

This example covers fitting a social preference model to count data
(where the count of social events per observation is recorded) with an
observation-level location effect, basic model checking and diagnostics,
visualising networks with uncertainty, calculating probability
distributions over network centrality, and propagating network
uncertainty into subsequent analyses.

# Setup

First of all we’ll load in Rstan for model fitting in Stan, dplyr for
handling the data, and igraph for network plotting and computing network
centrality. We also load in two custom R files: “simulations.R” to
generate synthetic data for this example; and “sampler.R” to allow
fitting models with uncertainty over network features respectively.

``` r
library(rstan)
library(dplyr)
library(igraph)

source("../scripts/simulations.R")
source("../scripts/sampler.R")
```

# Simulating data

Now we will simulate data using the `simulate_binary()` function. The
rows of the resulting dataframe describe observations at the dyadic
level between nodes. In this dataframe, `event` denotes whether or not
an undirected social event was observed in this observation period. The
exact definition of observation period will depend on the study, but is
commonly a sampling period where at least one of the members of the dyad
was observed. This can also be a sampling period where both members of
the dyad were observed, and the distinction will affect the
interpretation of social preferences. See the paper for further
discussion on this. `location` denotes the location at which the
observation took place, which may be relevant if location is likely to
impact the visibility of social events.

``` r
set.seed(1)
data <- simulate_count()
df <- data$df
head(df)
```

    ##   node_1  node_2   type_1   type_2 event_count location
    ## 1    Rey    Leia Lifeform Lifeform           3        C
    ## 2    Rey Obi-Wan Lifeform Lifeform           9        A
    ## 3    Rey Obi-Wan Lifeform Lifeform          26        F
    ## 4    Rey Obi-Wan Lifeform Lifeform           5        D
    ## 5    Rey Obi-Wan Lifeform Lifeform          13        A
    ## 6    Rey Obi-Wan Lifeform Lifeform          19        E

# Preparing the data

Computationally it’s easier to work with dyad IDs rather than pairs of
nodes in the statistical model, so we’ll map the pairs of nodes to dyad
IDs before we put the data into the model. The same is true for the
location factor, so we will also map the locations to location IDs. We
can add these columns to the dataframe using the following code:

``` r
df <- df %>%
  group_by(node_1, node_2) %>%
  mutate(dyad_id=cur_group_id()) %>%
  mutate(location_id=as.integer(location))
head(df)
```

    ## # A tibble: 6 × 8
    ## # Groups:   node_1, node_2 [2]
    ##   node_1 node_2  type_1   type_2   event_count location dyad_id location_id
    ##   <fct>  <fct>   <fct>    <fct>          <int> <fct>      <int>       <int>
    ## 1 Rey    Leia    Lifeform Lifeform           3 C              1           3
    ## 2 Rey    Obi-Wan Lifeform Lifeform           9 A              2           1
    ## 3 Rey    Obi-Wan Lifeform Lifeform          26 F              2           6
    ## 4 Rey    Obi-Wan Lifeform Lifeform           5 D              2           4
    ## 5 Rey    Obi-Wan Lifeform Lifeform          13 A              2           1
    ## 6 Rey    Obi-Wan Lifeform Lifeform          19 E              2           5

It will also be useful later to aggregate the dataframe at the dyad
level, assign dyad IDs corresponding to each dyad, and calculate total
event counts for each dyad. We can do this using:

``` r
df_agg <- df %>%
  group_by(node_1, node_2) %>%
  summarise(event_count_total=sum(event_count), dyad_id=cur_group_id()) %>%
  mutate(node_1_id=as.integer(node_1), node_2_id=as.integer(node_2))
head(df_agg)
```

    ## # A tibble: 6 × 6
    ## # Groups:   node_1 [1]
    ##   node_1 node_2  event_count_total dyad_id node_1_id node_2_id
    ##   <fct>  <fct>               <int>   <int>     <int>     <int>
    ## 1 Rey    Leia                    3       1         1         2
    ## 2 Rey    Obi-Wan               140       2         1         3
    ## 3 Rey    Luke                  169       3         1         4
    ## 4 Rey    C-3PO                  94       4         1         5
    ## 5 Rey    BB-8                  111       5         1         6
    ## 6 Rey    R2-D2                 241       6         1         7

Now we have all of the data in the right format for fitting the model,
we just need to put it into a list object. The data required by the
statistical model is defined in `binary_model.stan`.

``` r
model_data <- list(
  N=nrow(df), # Number of observations
  M=nrow(df_agg), # Number of dyads
  L=6, # Number of locations
  dyad_ids=df$dyad_id, # Vector of dyad IDs corresponding to each observation
  location_ids=df$location_id, # Vector of location IDs corresponding to each observation
  event_count=df$event_count # Vector of event counts corresponding to each observation
)
```

# Fitting the model

To fit the model, we first must compile it and load it into memory using
the function `stan_model()` and providing the filepath to the model. The
working directory will need to be set to the directory of the model for
this to work properly.

``` r
model <- stan_model("../models/count_model.stan")
```

Compiling the model may take a minute or two, but once this is done, the
model can be fit using `sampling()`. The argument `cores` sets the
number of CPU cores to be used for fitting the model, if your computer
has 4 or more cores, it’s worth setting this to 4.

``` r
fit <- sampling(model, model_data, cores=4, iter=5000, refresh=500)
```

# Model checking

The R-hat values provided by Stan indicate how well the chains have
converged, with values very close to 1.00 being ideal. Values diverging
from 1.00 indicate that the posterior samples may be very unreliable,
and shouldn’t be trusted. The chains can be plotted using Rstan’s
`traceplot` function to verify this visually:

``` r
traceplot(fit)
```

![](count_data_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

Good R-hat values don’t necessarily indicate that the model is
performing well, only that the parameter estimates appear to be robust.
To check that the model is performing as it should, a predictive check
can be used. A predictive check uses the fitted model to make
predictions, and compares those predictions to the observed data. The
predictions should indicate that the observed data are concordant with
the predictions from the model. There are many ways to perform a
predictive check, as data can be summarised in many different ways. For
the purposes of this example, we’ll use a simple density check where the
probability distributions of the aggregated event counts are compared
against the predictions from the model. Note that this isn’t a guarantee
that the model predictions are good, only that the predictions have the
same event count distribution as the data. Ideally several predictive
checks would be used to check the performance of the model.

This check uses predictions generated by the Stan model as the quantity
`event_pred`, with one set of predictions for each step in the MCMC
chain. The predictive check will randomly sample 10 of these steps,
compute the event counts for each dyad, and plot the densities against
the density of the observed event counts from the data.

``` r
# Extract event predictions from the fitted model
event_pred <- extract(fit)$event_pred
num_iterations <- dim(event_pred)[1]

# Plot the density of the observed event counts
plot(density(df_agg$event_count_total), main="", xlab="Dyadic event counts", ylim=c(0, 0.007))

# Plot the densities of the predicted event counts, repeat for 10 samples
df_copy <- df
for (i in 1:20) {
  df_copy$event_count <- event_pred[sample(1:num_iterations, size=1), ]
  df_agg_copy <- df_copy %>% 
    group_by(node_1, node_2) %>%
    summarise(event_count_total=sum(event_count))
  lines(density(df_agg_copy$event_count_total), col=rgb(0, 0, 1, 0.5))
}
```

![](count_data_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

This plot shows that the observed data falls well within the predicted
densities, and the predictions suggest the model has captured the main
features of the data well. Now we can be reasonably confident that the
model has fit correctly and describes the data well, so we can start to
make inferences from the model.

# Extracting social preferences

The main purpose of this part of the framework is to estimate social
preferences of dyads. We can access these using the `logit_p` quantity.
This will give a distribution of logit-scale social preferences for each
dyad, akin to an edge list. A more useful format for network data is
usually adjacency matrices, rather than edge lists, so instead we’ll
convert the distribution of edge lists to a distribution of adjacency
matrices, and store the result in an 8 x 8 x 4000 tensor, as there are 8
nodes and 4000 samples from the posterior.

``` r
log_p_samples <- extract(fit)$log_p

adj_tensor <- array(0, c(8, 8, num_iterations))
for (dyad_id in 1:model_data$M) {
  dyad_row <- df_agg[df_agg$dyad_id == dyad_id, ]
  adj_tensor[dyad_row$node_1_id, dyad_row$node_2_id, ] <- log_p_samples[, dyad_id]
}
adj_tensor[, , 1] # Print the first sample of the posterior distribution over adjacency matrices
```

    ##      [,1]      [,2]     [,3]     [,4]       [,5]      [,6]       [,7]
    ## [1,]    0 0.1788137 1.718366 1.431580  0.2810777 1.0019411  0.7784090
    ## [2,]    0 0.0000000 1.825351 1.552107 -0.7392504 0.8121894  0.8482851
    ## [3,]    0 0.0000000 0.000000 1.103999  0.9356539 0.9553636  0.5177972
    ## [4,]    0 0.0000000 0.000000 0.000000  0.4804553 0.5428505  1.0254778
    ## [5,]    0 0.0000000 0.000000 0.000000  0.0000000 0.5482965  0.3013480
    ## [6,]    0 0.0000000 0.000000 0.000000  0.0000000 0.0000000 -0.2133977
    ## [7,]    0 0.0000000 0.000000 0.000000  0.0000000 0.0000000  0.0000000
    ## [8,]    0 0.0000000 0.000000 0.000000  0.0000000 0.0000000  0.0000000
    ##             [,8]
    ## [1,] -1.92245104
    ## [2,]  0.40347286
    ## [3,]  0.68667214
    ## [4,]  0.33759101
    ## [5,]  1.05566661
    ## [6,] -0.07595785
    ## [7,]  0.26816728
    ## [8,]  0.00000000

The adjacency matrix above corresponds to a single draw of the posterior
adjacency matrices. You’ll notice that many of the entries are negative,
because the social preferences are on the logit scale. These can be
transformed back to the \[0, 1\] range using the logistic function. If
there are no additional effects (such as location in our case), the
transformed social preferences will be probabilities and the median will
be approximately the same as the simple ratio index for each dyad.
However, when additional effects are included, the transformed values
can no longer be interpreted as probabilities, though they may be useful
for visualisation and analysis purposes. We can logistic transform an
adjacency matrix using the logistic function (`plogis()` in base R).
This will also map 0 values to 0.5, so it will be necessary to set those
values back to zero again. This transformation can be achieved using the
following code:

``` r
plogis(adj_tensor[, , 1]) * upper.tri(adj_tensor[, , 1])
```

    ##      [,1]      [,2]      [,3]      [,4]      [,5]      [,6]      [,7]      [,8]
    ## [1,]    0 0.5445847 0.8479182 0.8071474 0.5698104 0.7314401 0.6853371 0.1275885
    ## [2,]    0 0.0000000 0.8612070 0.8252179 0.3231681 0.6925759 0.7002073 0.5995218
    ## [3,]    0 0.0000000 0.0000000 0.7510087 0.7182209 0.7221926 0.6266325 0.6652262
    ## [4,]    0 0.0000000 0.0000000 0.0000000 0.6178554 0.6324753 0.7360382 0.5836052
    ## [5,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.6337403 0.5747720 0.7418616
    ## [6,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.4468521 0.4810197
    ## [7,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.5666429
    ## [8,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000

It will be necessary to use this transformation for the visualisations
and analyses we have planned, so we’ll apply the transformation to the
entire tensor:

``` r
adj_tensor_transformed <- adj_tensor
for (i in 1:dim(adj_tensor)[3]) {
  adj_tensor_transformed[, , i] <- plogis(adj_tensor[, , i]) * upper.tri(adj_tensor[, , i])
}
```

# Visualising uncertainty

The aim of our network visualisation is to plot a network where the
certainty in social preferences (edge weights) can be seen. To do this
we’ll use a semi-transparent line around each edge with a width that
corresponds to a standardised uncertainty measures. The uncertainty
measure will simply be the normalised difference between the 97.5% and
2.5% credible interval estimate for each social preference. We can
calculate this from the transformed adjacency tensor object, generate
two igraph objects for the main network and the uncertainty in edges,
and plot them with the same coordinates.

``` r
# Calculate lower, median, and upper quantiles of edge weights. Lower and upper give credible intervals.
adj_quantiles <- apply(adj_tensor_transformed, c(1, 2), function(x) quantile(x, probs=c(0.025, 0.5, 0.975)))
adj_lower <- adj_quantiles[1, , ]
adj_mid <- adj_quantiles[2, , ]
adj_upper <- adj_quantiles[3, , ]

# Calculate standardised width/range of credible intervals.
adj_range <- ((adj_upper - adj_lower)/adj_mid)
adj_range[is.nan(adj_range)] <- 0

# Generate two igraph objects, one form the median and one from the standardised width.
g_mid <- graph_from_adjacency_matrix(adj_mid, mode="undirected", weighted=TRUE)
g_range <- graph_from_adjacency_matrix(adj_range, mode="undirected", weighted=TRUE)

# Plot the median graph first and then the standardised width graph to show uncertainty over edges.
coords <- igraph::layout_nicely(g_mid)
plot(g_mid, edge.width=3 * E(g_mid)$weight, edge.color="black",  layout=coords)
plot(g_mid, edge.width=2 * 3 * E(g_range)$weight, edge.color=rgb(0, 0, 0, 0.25), 
     vertex.label=c("Rey", "Leia", "Obi-Wan", "Luke", "C-3PO", "BB-8", "R2-D2", "D-O"), 
     vertex.label.dist=4, vertex.label.color="black", layout=coords, add=TRUE)
```

![](count_data_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

This plot can be extended in multiple ways, for example by thresholding
low edge weights to visualise the network more tidily, or by adding
halos around nodes to show uncertainty around network centrality, and so
on.

# Extracting network centralities

Uncertainty around network metrics such as centrality can be calculated
quite simply by drawing adjacency matrices from the posterior
distribution over adjacency matrices, generating a network from them,
and calculating the network metric of interest. It is important to
sample over the adjacency matrices rather than by the edges on their
own, as this maintains the joint distribution of edge weights and will
generate more reliable and accurate estimates of network centrality.

``` r
centrality_matrix <- matrix(0, nrow=num_iterations, ncol=8)
for (i in 1:num_iterations) {
  g <- graph_from_adjacency_matrix(adj_tensor[, , i], mode="undirected", weighted=TRUE)
  centrality_matrix[i, ] <- strength(g)
}
colnames(centrality_matrix) <- c("Rey", "Leia", "Obi-Wan", "Luke", "C-3PO", "BB-8", "R2-D2", "D-O")
head(centrality_matrix)
```

    ##            Rey      Leia   Obi-Wan      Luke    C-3PO     BB-8    R2-D2
    ## [1,]  5.390187  5.620220  7.743204  6.474061 3.602498 3.860641 3.739484
    ## [2,]  8.889050  9.506583  9.496294  8.914212 5.387568 6.246322 5.928575
    ## [3,] 10.125602 10.387921 11.597704 10.506199 6.836469 7.734962 7.040915
    ## [4,]  5.082064  5.511146  6.272643  4.595521 2.167484 2.455867 2.338378
    ## [5,]  6.664884  7.416372  7.771342  6.605749 3.958090 4.202894 3.888913
    ## [6,]  7.142724  7.424169  8.277317  7.776276 4.401275 4.957974 4.436312
    ##           D-O
    ## [1,] 2.751570
    ## [2,] 4.575158
    ## [3,] 5.886973
    ## [4,] 1.557004
    ## [5,] 2.849120
    ## [6,] 3.770041

Each column in this matrix corresponds to one of the nodes in the
network, and each row is its centrality in one sample of the posterior
distribution of the adjacency matrices. We can calculate the credible
intervals using the `quantile` function as follows:

``` r
centrality_quantiles <- t(apply(centrality_matrix, 2, function(x) quantile(x, probs=c(0.025, 0.5, 0.975))))
centrality_quantiles
```

    ##              2.5%      50%     97.5%
    ## Rey     3.4286986 7.867386 11.375457
    ## Leia    3.7315403 8.291435 11.937091
    ## Obi-Wan 4.1597053 9.419373 13.235410
    ## Luke    3.3762768 8.523182 12.406659
    ## C-3PO   1.0377673 5.053229  8.544627
    ## BB-8    1.4964563 5.501766  9.255069
    ## R2-D2   1.3146507 5.216039  8.932726
    ## D-O     0.6929897 4.140116  7.549190

# Maintaining uncertainty in regression on centralities

The key challenge to quantifying uncertainty in network analysis is to
incorporate uncertainty due to sampling into downstream analyses,
commonly regression. This can be achieved by modifying the likelihood
function of a regression model to treat the network centralities with
uncertainty. We have written a custom MCMC sampler function that samples
from the joint distribution of network centralities calculated earlier
and treats those samples as the data in the likelihood function.
Likelihood functions for the sampler use the `index` variable to keep
track of which data points are being compared internally in the sampler,
to ensure that candidate steps in the MCMC are not accepted or rejected
because they are being compared to different data points, rather than
because of the parameter space.

Custom likelihood functions take a similar form to the `target +=`
syntax in Stan, but for more specific resources the following document
is a good start: <https://www.ime.unicamp.br/~cnaber/optim_1.pdf>. We
will implement a linear regression to test if lifeforms are more central
in the social network than droids. We have included a coefficient for
both lifeform and droid, unlike standard frequentist models. This is
because using a reference category (such as droid) would imply that
there is less uncertainty around the centrality of droids than around
lifeforms. It also allows for easy comparison between categories by
calculating the difference in posteriors.

``` r
loglik <- function(params, Y, X, index) {
  # Define parameters
  intercept <- params[1]
  beta_lifeform <- params[2]
  beta_droid <- params[3]
  sigma <- exp(params[4]) # Exponential keeps underlying value unconstrained, which is much easier for the sampler.
  
  # Sample data according to index
  y <- Y[index %% dim(Y)[1] + 1, ]
  
  # Define model
  target <- 0
  target <- target + sum(dnorm(y, mean=intercept + beta_lifeform * X[, 1] + beta_droid * X[, 2], sd=sigma, log=TRUE)) # Main model
  target <- target + dnorm(intercept, mean=0, sd=2.5, log=TRUE) # Prior on intercept
  target <- target + dnorm(beta_lifeform, mean=0, sd=2.5, log=TRUE) # Prior on lifeform coefficient
  target <- target + dnorm(beta_droid, mean=0, sd=2.5, log=TRUE) # Prior on droid coefficient
  target <- target + dexp(sigma, 1, log=TRUE) # Prior on sigma
  
  return(target)
}
```

Now we will prepare data for fitting the model. The predictor matrix is
simply a matrix with 2 columns and 8 rows, corresponding to whether each
of the 8 nodes is a lifeform (column 1) or a droid (column 2).

``` r
predictor_matrix <- matrix(0, nrow=8, ncol=2)
colnames(predictor_matrix) <- c("lifeform", "droid")
predictor_matrix[1:4, 1] <- 1
predictor_matrix[5:8, 2] <- 1
predictor_matrix
```

    ##      lifeform droid
    ## [1,]        1     0
    ## [2,]        1     0
    ## [3,]        1     0
    ## [4,]        1     0
    ## [5,]        0     1
    ## [6,]        0     1
    ## [7,]        0     1
    ## [8,]        0     1

Since network strength is strictly positive, a Gaussian error is not a
reasonable model for the data. The Gaussian family model is much easier
to implement as well as interpret than many other models, so we will
standardise the centralities by taking z-scores.

``` r
centrality_matrix_std <- (centrality_matrix - apply(centrality_matrix, 1, mean))/apply(centrality_matrix, 1, sd)
centrality_matrix_std[is.nan(centrality_matrix_std)] <-0
head(centrality_matrix_std)
```

    ##            Rey      Leia  Obi-Wan      Luke      C-3PO       BB-8      R2-D2
    ## [1,] 0.2909999 0.4269299 1.681439 0.9314799 -0.7653772 -0.6128359 -0.6844296
    ## [2,] 0.7491261 1.0532587 1.048192 0.7615182 -0.9753408 -0.5524076 -0.7088969
    ## [3,] 0.6408051 0.7643127 1.333916 0.8200018 -0.9078203 -0.4847820 -0.8115608
    ## [4,] 0.7372732 0.9743198 1.395009 0.4684824 -0.8728881 -0.7135706 -0.7784777
    ## [5,] 0.6583495 1.0556639 1.243338 0.6270843 -0.7727435 -0.6433144 -0.8093175
    ## [6,] 0.6212305 0.7774146 1.250857 0.9728116 -0.9000988 -0.5911661 -0.8806555
    ##            D-O
    ## [1,] -1.268206
    ## [2,] -1.375450
    ## [3,] -1.354872
    ## [4,] -1.210148
    ## [5,] -1.359060
    ## [6,] -1.250394

Now we’re in a position to fit the model. To do this, we define the
target function, which is simply a function that maps candidate
parameters and a network centrality index to the log-likelihood of that
function for the given sample of the centrality posterior. This means
the target function can be written as a function of the data
`centrality_matrix_std` and `predictor_matrix`.

``` r
target <- function(params, index) loglik(params, centrality_matrix_std, predictor_matrix, index)
```

The function `metropolis` from `sampler.R` can now be used to fit the
model using the provided target function, an initial set of parameters,
and some additional MCMC options.

``` r
chain <- metropolis(target, c(0, 0, 0, 0), iterations=100000, thin=100, refresh=10000)
```

    ## Chain: 1 | Iteration: 10000/102000 (Sampling)
    ## Chain: 1 | Iteration: 20000/102000 (Sampling)
    ## Chain: 1 | Iteration: 30000/102000 (Sampling)
    ## Chain: 1 | Iteration: 40000/102000 (Sampling)
    ## Chain: 1 | Iteration: 50000/102000 (Sampling)
    ## Chain: 1 | Iteration: 60000/102000 (Sampling)
    ## Chain: 1 | Iteration: 70000/102000 (Sampling)
    ## Chain: 1 | Iteration: 80000/102000 (Sampling)
    ## Chain: 1 | Iteration: 90000/102000 (Sampling)
    ## Chain: 1 | Iteration: 100000/102000 (Sampling)
    ## Acceptance Rate: 0.232372549019608

``` r
colnames(chain) <- c("intercept", "beta_lifeform", "beta_droid", "sigma")
head(chain)
```

    ##      intercept beta_lifeform beta_droid      sigma
    ## [1,] 0.4433066    0.54492267  -1.244671 -0.9633308
    ## [2,] 0.2343876    0.50582838  -1.289113 -1.6179369
    ## [3,] 0.1237907    0.70058411  -1.486700 -0.6269339
    ## [4,] 0.9071564    0.06641298  -1.817585 -1.0491932
    ## [5,] 1.4363816   -0.78373346  -2.573370 -0.9716079
    ## [6,] 1.7712614   -0.77720613  -2.744187 -0.8317323

# Checking the regression

The resulting chain of MCMC samples forms the posterior distribution of
parameter estimates for the regression model. But before we look at
these too closely, we should check that the chains have converged:

``` r
par(mfrow=c(2, 2))
for (i in 1:4) {
  plot(chain[, i], type="l")
}
```

![](count_data_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

These chains appear to be quite healthy. Ideally we would run multiple
additional chains starting at different points to check that they
converge and mix properly. For the sake of this example we won’t go into
that here.

Again, the performance of the sampler doesn’t necessarily guarantee the
performance of the model, so we’ll use predictive checks to test the
performance of the model. In this case, the data are not fixed, and
there are multiple possible values they can take. Therefore we’ll plot
the distribution of centrality values on different draws of the
adjacency matrices as well as the distribution of predicted centrality
values on different draws.

``` r
plot(density(centrality_matrix_std[1, ]), ylim=c(0, 0.7), main="", xlab="Standardised network strength")
sample_ids <- sample(1:1000, size=200)
for (i in sample_ids) {
  pred <- rnorm(8, mean=chain[i, "intercept"] + chain[i, "beta_lifeform"] * predictor_matrix[, 1] + chain[i, "beta_droid"] * predictor_matrix[, 2], sd=exp(chain[i, "sigma"]))
  lines(density(centrality_matrix_std[i, ]), col=rgb(0, 0, 0, 0.25))
  lines(density(pred), col=rgb(0, 0, 1, 0.25))
}
```

![](count_data_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

The model appears to fit reasonably well, and the observed data are
completely consistent with the predictions of the model, so we can go
ahead with the analysis.

# Interpreting the regression

The regression coefficients and parameters can be summarised by
calculating their percentile credible interval similar to before:

``` r
coefficient_quantiles <- t(apply(chain, 2, function(x) quantile(x, probs=c(0.025, 0.5, 0.975))))
coefficient_quantiles
```

    ##                    2.5%        50%      97.5%
    ## intercept     -2.619299  0.1162728  2.8267462
    ## beta_lifeform -1.904701  0.7803286  3.5330472
    ## beta_droid    -3.588326 -0.9855091  1.6807389
    ## sigma         -1.502878 -0.9768640 -0.3148773

A frequentist analysis (and some Bayesian ones too) would have only one
category, lifeform or droid, and the other category would be the
implicit reference category, absorbed by the intercept. In this type of
analysis, the coefficients for the two categories correspond to the
average difference between the centrality of nodes in that category
compared to the population average (the intercept). Therefore, to look
for a difference between the two categories, we can simply calculate the
difference in the posterior distributions of those two categories:

``` r
beta_difference <- chain[, "beta_lifeform"] - chain[, "beta_droid"]
quantile(beta_difference, probs=c(0.025, 0.5, 0.975))
```

    ##     2.5%      50%    97.5% 
    ## 1.133950 1.750348 2.321770

The mass of probability is with there being a positive difference of
around 1.57 standard deviations between the centralities of lifeforms
compared to droids. Many of the benefits of Bayesian analysis only apply
when significance testing is avoided. Though it is reasonably common for
a result such as the one above not overlapping zero to be interpreted as
being “significant”, using such a decision rule leaves Bayesian analysis
open to the same flaws as frequentist analyses often have. For this
reason we caution strongly against using such a rule.

# Conclusion

In this guide we have shown how to apply the social preference model to
binary presence/absence data and how to conduct subsequent analyses,
while maintaining uncertainty through the whole process. Though this
process is quite hands-on, it provides a huge amount of flexibility for
conducting animal social network analyses in a robust and interpretable
way.
