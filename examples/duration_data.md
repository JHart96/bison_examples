Duration Data Example
================

This example covers fitting a edge weight model to duration data (where
the duration of social events is recorded) with an observation-level
location effect, basic model checking and diagnostics, visualising
networks with uncertainty, calculating probability distributions over
network centrality, and propagating network uncertainty into subsequent
analyses.

*Note: Many of the procedures presented here are stochastic, and plots
and results may vary between compilations of this document. In
particular, MCMC chains and model estimates may sometimes not be
optimal, even if they were when it was originally written.*

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

Now we will simulate data using the `simulate_duration()` function. The
rows of the resulting dataframe describe observations at the dyadic
level between nodes. In this dataframe, `event` denotes whether or not
an undirected social event was observed in this observation period, and
`duration` denotes the length of such an event. The exact definition of
observation period will depend on the study, but is commonly a sampling
period where at least one of the members of the dyad was observed. This
can also be a sampling period where both members of the dyad were
observed, and the distinction will affect the interpretation of edge
weights. See the paper for further discussion on this. `location`
denotes the location at which the observation took place, which may be
relevant if location is likely to impact the visibility of social
events.

``` r
set.seed(1)
data <- simulate_duration()
```

    ## `summarise()` has grouped output by 'node_1'. You can override using the
    ## `.groups` argument.

``` r
df_obs <- data$df_obs
df_obs_agg <- data$df_obs_agg
head(df_obs)
```

    ##   node_1 node_2 duration event location
    ## 1    Rey   Leia      600     1        A
    ## 2    Rey   Leia       42     1        A
    ## 3    Rey   Leia       35     1        D
    ## 4    Rey   Leia       68     1        E
    ## 5    Rey   Leia        3     1        B
    ## 6    Rey   Leia       59     1        D

``` r
head(df_obs_agg)
```

    ## # A tibble: 6 × 7
    ## # Groups:   node_1 [1]
    ##   node_1 node_2  total_event_time num_events total_obs_time node_1_type
    ##   <fct>  <fct>              <dbl>      <dbl>          <dbl> <fct>      
    ## 1 Rey    Leia               12833        114          28800 Lifeform   
    ## 2 Rey    Obi-Wan             5620         87          28200 Lifeform   
    ## 3 Rey    Luke                7788         66          26400 Lifeform   
    ## 4 Rey    C-3PO               1174         51          24000 Lifeform   
    ## 5 Rey    BB-8                 699         59          30000 Lifeform   
    ## 6 Rey    R2-D2                860         13          25800 Lifeform   
    ## # … with 1 more variable: node_2_type <fct>

In our simulated data, `total_event_time` is the total duration of
social events, `num_events` is the number of social events, and
`total_obs_time` is the total duration of observations.

# Preparing the data

Computationally it’s easier to work with dyad IDs rather than pairs of
nodes in the statistical model, so we’ll map the pairs of nodes to dyad
IDs before we put the data into the model. The same is true for the
location factor, so we will also map the locations to location IDs. We
can add these columns to the dataframe using the following code:

``` r
df_obs <- df_obs %>%
  group_by(node_1, node_2) %>%
  mutate(dyad_id=cur_group_id()) %>%
  mutate(location_id=as.integer(location), location_id=ifelse(is.na(location_id), 0, location_id))
head(df_obs)
```

    ## # A tibble: 6 × 7
    ## # Groups:   node_1, node_2 [1]
    ##   node_1 node_2 duration event location dyad_id location_id
    ##   <fct>  <fct>     <dbl> <dbl> <fct>      <int>       <int>
    ## 1 Rey    Leia        600     1 A              1           1
    ## 2 Rey    Leia         42     1 A              1           1
    ## 3 Rey    Leia         35     1 D              1           4
    ## 4 Rey    Leia         68     1 E              1           5
    ## 5 Rey    Leia          3     1 B              1           2
    ## 6 Rey    Leia         59     1 D              1           4

It will also be useful later to aggregate the dataframe at the dyad
level and assign dyad IDs corresponding to each dyad. We can do this
using:

``` r
df_obs_agg <- df_obs_agg %>%
  group_by(node_1, node_2) %>%
  mutate(dyad_id=cur_group_id(), node_1_id=as.integer(node_1), node_2_id=as.integer(node_2))
head(df_obs_agg)
```

    ## # A tibble: 6 × 10
    ## # Groups:   node_1, node_2 [6]
    ##   node_1 node_2  total_event_time num_events total_obs_time node_1_type
    ##   <fct>  <fct>              <dbl>      <dbl>          <dbl> <fct>      
    ## 1 Rey    Leia               12833        114          28800 Lifeform   
    ## 2 Rey    Obi-Wan             5620         87          28200 Lifeform   
    ## 3 Rey    Luke                7788         66          26400 Lifeform   
    ## 4 Rey    C-3PO               1174         51          24000 Lifeform   
    ## 5 Rey    BB-8                 699         59          30000 Lifeform   
    ## 6 Rey    R2-D2                860         13          25800 Lifeform   
    ## # … with 4 more variables: node_2_type <fct>, dyad_id <int>, node_1_id <int>,
    ## #   node_2_id <int>

Now we have all of the data in the right format for fitting the model,
we just need to put it into a list object. The data required by the
statistical model is defined in `duration_model.stan`.

``` r
model_data <- list(
  N=nrow(df_obs), # Number of observations
  M=nrow(df_obs_agg), # Number of dyads
  L=6, # Number of locations
  dyad_ids=df_obs$dyad_id, # Vector of dyad IDs corresponding to each observation
  location_ids=df_obs$location_id, # Vector of location IDs corresponding to each observation
  durations=df_obs$duration, # Vector of event durations corresponding to each observation
  num_events=df_obs_agg$num_events, # Vector of event counts corresponding to each dyad
  total_obs_time=df_obs_agg$total_event_time # Vector of total observation times corresponding to each dyad
)
```

# Fitting the model

To fit the model, we first must compile it and load it into memory using
the function `stan_model()` and providing the filepath to the model. The
working directory will need to be set to the directory of the model for
this to work properly.

``` r
model <- stan_model("../models/duration_model.stan")
```

Compiling the model may take a minute or two, but once this is done, the
model can be fit using `sampling()`. The argument `cores` sets the
number of CPU cores to be used for fitting the model, if your computer
has 4 or more cores, it’s worth setting this to 4.

``` r
fit <- sampling(model, model_data, cores=4)
```

    ## Warning: Bulk Effective Samples Size (ESS) is too low, indicating posterior means and medians may be unreliable.
    ## Running the chains for more iterations may help. See
    ## http://mc-stan.org/misc/warnings.html#bulk-ess

# Model checking

The R-hat values provided by Stan indicate how well the chains have
converged, with values very close to 1.00 being ideal. Values diverging
from 1.00 indicate that the posterior samples may be very unreliable,
and shouldn’t be trusted. The chains can be plotted using Rstan’s
`traceplot` function to verify this visually:

``` r
traceplot(fit)
```

![](duration_data_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

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
plot(density(df_obs$duration), main="", xlab="Dyadic event durations", ylim=c(0, 0.04))

# Plot the densities of the predicted event counts, repeat for 10 samples
df_copy <- df
for (i in 1:20) {
  event_durations_pred <- event_pred[sample(1:num_iterations, size=1), ]
  lines(density(event_durations_pred), col=rgb(0, 0, 1, 0.5))
}
```

![](duration_data_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

This plot shows that the observed data falls well within the predicted
densities, and the predictions suggest the model has captured the main
features of the data well. Now we can be reasonably confident that the
model has fit correctly and describes the data well, so we can start to
make inferences from the model.

# Extracting edge weights

The main purpose of this part of the framework is to estimate edge
weights of dyads. We can access these using the `logit_p` quantity. This
will give a distribution of logit-scale edge weights for each dyad, akin
to an edge list. A more useful format for network data is usually
adjacency matrices, rather than edge lists, so instead we’ll convert the
distribution of edge lists to a distribution of adjacency matrices, and
store the result in an 8 x 8 x 4000 tensor, as there are 8 nodes and
4000 samples from the posterior.

``` r
logit_p_samples <- extract(fit)$logit_p

adj_tensor <- array(0, c(8, 8, num_iterations))
for (dyad_id in 1:model_data$M) {
  dyad_row <- df_obs_agg[df_obs_agg$dyad_id == dyad_id, ]
  adj_tensor[dyad_row$node_1_id, dyad_row$node_2_id, ] <- logit_p_samples[, dyad_id]
}
adj_tensor[, , 1] # Print the first sample of the posterior distribution over adjacency matrices
```

    ##      [,1]     [,2]     [,3]      [,4]        [,5]       [,6]       [,7]
    ## [1,]    0 1.352989 1.436333 1.8010792 -0.01140462 -0.4667570 -0.7766221
    ## [2,]    0 0.000000 2.332313 0.6037593 -1.13781425 -0.5125763  0.5097058
    ## [3,]    0 0.000000 0.000000 1.9091110 -0.90578464 -0.6189982 -0.6351699
    ## [4,]    0 0.000000 0.000000 0.0000000 -0.68035050 -0.3011211 -0.8958456
    ## [5,]    0 0.000000 0.000000 0.0000000  0.00000000 -0.4164682  1.5528491
    ## [6,]    0 0.000000 0.000000 0.0000000  0.00000000  0.0000000 -0.1199744
    ## [7,]    0 0.000000 0.000000 0.0000000  0.00000000  0.0000000  0.0000000
    ## [8,]    0 0.000000 0.000000 0.0000000  0.00000000  0.0000000  0.0000000
    ##            [,8]
    ## [1,] -0.8162401
    ## [2,] -1.3894768
    ## [3,]  0.2742975
    ## [4,]  0.7757955
    ## [5,]  1.8077165
    ## [6,] -1.0915465
    ## [7,]  0.5741771
    ## [8,]  0.0000000

The adjacency matrix above corresponds to a single draw of the posterior
adjacency matrices. You’ll notice that many of the entries are negative,
because the edge weights are on the logit scale. These can be
transformed back to the \[0, 1\] range using the logistic function. If
there are no additional effects (such as location in our case), the
transformed edge weights will be probabilities and the median will be
approximately the same as the simple ratio index for each dyad. However,
when additional effects are included, the transformed values can no
longer be interpreted as probabilities, though they may be useful for
visualisation and analysis purposes. We can logistic transform an
adjacency matrix using the logistic function (`plogis()` in base R).
This will also map 0 values to 0.5, so it will be necessary to set those
values back to zero again. This transformation can be achieved using the
following code:

``` r
plogis(adj_tensor[, , 1]) * upper.tri(adj_tensor[, , 1])
```

    ##      [,1]      [,2]      [,3]      [,4]      [,5]      [,6]      [,7]      [,8]
    ## [1,]    0 0.7946179 0.8078862 0.8582803 0.4971489 0.3853841 0.3150484 0.3065624
    ## [2,]    0 0.0000000 0.9115180 0.6465159 0.2427219 0.3745898 0.6247375 0.1994913
    ## [3,]    0 0.0000000 0.0000000 0.8709192 0.2878632 0.3500093 0.3463392 0.5681476
    ## [4,]    0 0.0000000 0.0000000 0.0000000 0.3361831 0.4252834 0.2899050 0.6847732
    ## [5,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.3973622 0.8253249 0.8590857
    ## [6,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.4700423 0.2513272
    ## [7,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.6397265
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
certainty in edge weights (edge weights) can be seen. To do this we’ll
use a semi-transparent line around each edge with a width that
corresponds to a standardised uncertainty measures. The uncertainty
measure will simply be the normalised difference between the 97.5% and
2.5% credible interval estimate for each edge weight. We can calculate
this from the transformed adjacency tensor object, generate two igraph
objects for the main network and the uncertainty in edges, and plot them
with the same coordinates.

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

![](duration_data_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

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

    ##            Rey      Leia   Obi-Wan     Luke    C-3PO     BB-8     R2-D2
    ## [1,]  4.590402  4.798767  5.952054 5.089745 3.360566 0.000000 2.6367320
    ## [2,] 13.820179 11.061748 12.788068 8.694140 5.998521 2.538344 8.7081275
    ## [3,]  4.609520  5.333928  5.104937 3.797112 1.369595 0.000000 1.3122495
    ## [4,]  4.287150  4.604827  4.423657 4.762505 2.465043 0.000000 2.4394709
    ## [5,]  3.440103  4.431255  3.177518 2.659386 1.055953 0.000000 0.9430346
    ## [6,]  6.104716  5.856583  5.348248 5.596898 2.403255 0.000000 2.0057466
    ##           D-O
    ## [1,] 3.431987
    ## [2,] 7.387684
    ## [3,] 1.988893
    ## [4,] 3.225703
    ## [5,] 1.427429
    ## [6,] 3.642094

Each column in this matrix corresponds to one of the nodes in the
network, and each row is its centrality in one sample of the posterior
distribution of the adjacency matrices. We can calculate the credible
intervals using the `quantile` function as follows:

``` r
centrality_quantiles <- t(apply(centrality_matrix, 2, function(x) quantile(x, probs=c(0.025, 0.5, 0.975))))
centrality_quantiles
```

    ##               2.5%      50%     97.5%
    ## Rey     1.93750227 5.101661 10.128428
    ## Leia    1.40342796 5.147526  9.912717
    ## Obi-Wan 1.59669981 5.494188 10.817365
    ## Luke    1.39861638 4.883407  9.189791
    ## C-3PO   0.13808882 2.129185  5.611686
    ## BB-8    0.00000000 0.000000  2.301269
    ## R2-D2   0.04571567 2.791299  8.123343
    ## D-O     0.30683324 3.019667  6.949016

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

    ##            Rey      Leia   Obi-Wan        Luke      C-3PO      BB-8       R2-D2
    ## [1,] 0.4632801 0.5758047 1.1986203  0.73294291 -0.2008747 -2.015698 -0.59177027
    ## [2,] 1.3407147 0.5929215 1.0609159 -0.04892189 -0.7796872 -1.717719 -0.04513006
    ## [3,] 0.8261364 1.1844977 1.0712166  0.42424196 -0.7766388 -1.454171 -0.80500756
    ## [4,] 0.6235228 0.8194255 0.7077026  0.91666172 -0.5001234 -2.020252 -0.51589321
    ## [5,] 0.8592192 1.5151836 0.6854359  0.34252555 -0.7186580 -1.417509 -0.79338980
    ## [6,] 1.0029276 0.8915819 0.6634757  0.77505279 -0.6580381 -1.736456 -0.83641280
    ##              D-O
    ## [1,] -0.16230486
    ## [2,] -0.40309378
    ## [3,] -0.47027524
    ## [4,] -0.03104436
    ## [5,] -0.47280775
    ## [6,] -0.10213089

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
    ## Acceptance Rate: 0.228519607843137

``` r
colnames(chain) <- c("intercept", "beta_lifeform", "beta_droid", "sigma")
head(chain)
```

    ##       intercept beta_lifeform beta_droid       sigma
    ## [1,] -0.3231046     0.6481360 -0.3221333 -0.54718967
    ## [2,] -1.1113709     0.8159387  0.7257524  0.08682922
    ## [3,] -0.7227057     1.9666992  0.4234891 -0.18172689
    ## [4,]  0.4691291     0.2564554 -1.5031375 -0.72811267
    ## [5,]  1.0173916    -0.2139330 -1.7129112 -0.40052833
    ## [6,]  0.4708770     0.9484775 -1.6291662 -0.72317784

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

![](duration_data_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

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

![](duration_data_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

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

    ##                     2.5%         50%     97.5%
    ## intercept     -3.5300562  0.04018068 3.1322534
    ## beta_lifeform -2.4406386  0.70584853 4.2344502
    ## beta_droid    -3.8856530 -0.73335042 2.9830723
    ## sigma         -0.9155959 -0.37137835 0.3065001

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

    ##      2.5%       50%     97.5% 
    ## 0.2028548 1.4391780 2.5209736

The mass of probability is with there being a positive difference of
around 1.57 standard deviations between the centralities of lifeforms
compared to droids. Many of the benefits of Bayesian analysis only apply
when significance testing is avoided. Though it is reasonably common for
a result such as the one above not overlapping zero to be interpreted as
being “significant”, using such a decision rule leaves Bayesian analysis
open to the same flaws as frequentist analyses often have. For this
reason we caution strongly against using such a rule.

# Conclusion

In this guide we have shown how to apply the edge weight model to
duration data and how to conduct subsequent analyses, while maintaining
uncertainty through the whole process. Though this process is quite
hands-on, it provides a huge amount of flexibility for conducting animal
social network analyses in a robust and interpretable way.
