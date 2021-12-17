Duration Data Example
================

This example covers fitting a social preference model to duration data
(where the duration of social events is recorded) with an
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

Now we will simulate data using the `simulate_duration()` function. The
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

total_event_time is the total duration of social events, num_events is
the number of social events, and total_obs_time is the total duration of
observations.

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

It will also be useful later to aggregate the dataframe at the dyad
level, assign dyad IDs corresponding to each dyad, and calculate total
event counts for each dyad. We can do this using:

Now we have all of the data in the right format for fitting the model,
we just need to put it into a list object. The data required by the
statistical model is defined in `binary_model.stan`.

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
df_obs_agg$both_lifeform <- as.integer(df_obs_agg$node_1_type == "Lifeform" & df_obs_agg$node_2_type == "Lifeform")
```

``` r
logit_p_samples <- extract(fit)$logit_p

adj_tensor <- array(0, c(8, 8, num_iterations))
for (dyad_id in 1:model_data$M) {
  dyad_row <- df_obs_agg[df_obs_agg$dyad_id == dyad_id, ]
  adj_tensor[dyad_row$node_1_id, dyad_row$node_2_id, ] <- logit_p_samples[, dyad_id]
}
adj_tensor[, , 1] # Print the first sample of the posterior distribution over adjacency matrices
```

    ##      [,1]     [,2]     [,3]      [,4]       [,5]        [,6]        [,7]
    ## [1,]    0 1.545145 1.402663 1.4073912  0.1272230 -0.02903827 -0.71657439
    ## [2,]    0 0.000000 3.830558 0.6439938 -0.5034262 -0.49685602  0.46257351
    ## [3,]    0 0.000000 0.000000 1.0163840 -0.5202109 -0.04456489 -0.13936653
    ## [4,]    0 0.000000 0.000000 0.0000000 -0.4669158 -0.23020877 -1.14356223
    ## [5,]    0 0.000000 0.000000 0.0000000  0.0000000 -0.78570981  1.62677143
    ## [6,]    0 0.000000 0.000000 0.0000000  0.0000000  0.00000000 -0.01281999
    ## [7,]    0 0.000000 0.000000 0.0000000  0.0000000  0.00000000  0.00000000
    ## [8,]    0 0.000000 0.000000 0.0000000  0.0000000  0.00000000  0.00000000
    ##            [,8]
    ## [1,] -0.6830763
    ## [2,] -1.3457208
    ## [3,]  0.4378401
    ## [4,]  2.1857453
    ## [5,]  1.3831669
    ## [6,] -0.9686341
    ## [7,]  0.8095666
    ## [8,]  0.0000000

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
    ## [1,]    0 0.8242115 0.8026062 0.8033541 0.5317629 0.4927409 0.3281478 0.3355750
    ## [2,]    0 0.0000000 0.9787633 0.6556557 0.3767358 0.3782798 0.6136245 0.2065709
    ## [3,]    0 0.0000000 0.0000000 0.7342677 0.3728029 0.4888606 0.4652147 0.6077442
    ## [4,]    0 0.0000000 0.0000000 0.0000000 0.3853465 0.4427006 0.2416669 0.8989621
    ## [5,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.3130906 0.8357269 0.7994991
    ## [6,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.4967950 0.2751528
    ## [7,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.6920171
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

![](duration_data_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

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

    ##           Rey     Leia  Obi-Wan     Luke     C-3PO     BB-8     R2-D2      D-O
    ## [1,] 4.482423 6.482271 6.687446 5.253514 3.1371613 0.000000 2.8989115 4.816319
    ## [2,] 5.693434 6.396063 4.807799 4.373306 2.4774673 0.000000 3.2736776 4.424661
    ## [3,] 1.652067 1.476217 1.887461 1.195832 0.8220648 0.000000 0.1265214 1.034265
    ## [4,] 3.163382 3.101469 4.043083 3.640222 1.5744981 0.000000 0.2211680 2.297231
    ## [5,] 2.822196 2.588310 2.706548 3.087931 0.3120669 0.000000 1.5069754 2.531554
    ## [6,] 6.714511 5.656415 8.604494 8.154194 5.2301920 1.123028 4.6441589 3.879093

Each column in this matrix corresponds to one of the nodes in the
network, and each row is its centrality in one sample of the posterior
distribution of the adjacency matrices. We can calculate the credible
intervals using the `quantile` function as follows:

``` r
centrality_quantiles <- t(apply(centrality_matrix, 2, function(x) quantile(x, probs=c(0.025, 0.5, 0.975))))
centrality_quantiles
```

    ##               2.5%      50%     97.5%
    ## Rey     2.07656414 5.082967  9.647479
    ## Leia    1.52930726 5.103702  9.256464
    ## Obi-Wan 1.84496940 5.404357 10.153049
    ## Luke    1.49043206 4.890935  8.738310
    ## C-3PO   0.15809290 2.107845  5.214322
    ## BB-8    0.00000000 0.000000  1.838544
    ## R2-D2   0.06097449 2.765157  7.497973
    ## D-O     0.35317864 2.989039  6.557842

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

    ##            Rey       Leia   Obi-Wan      Luke      C-3PO      BB-8      R2-D2
    ## [1,] 0.1202040 1.03539232 1.1292861 0.4730778 -0.4954264 -1.931082 -0.6044564
    ## [2,] 0.8752666 1.22416967 0.4354888 0.2197338 -0.7216786 -1.951909 -0.3263062
    ## [3,] 0.9186262 0.66129835 1.2630849 0.2510027 -0.2959422 -1.498895 -1.3137522
    ## [4,] 0.5943934 0.55387487 1.1701028 0.9064557 -0.4454323 -1.475843 -1.3311027
    ## [5,] 0.7323670 0.53721884 0.6358734 0.9540881 -1.3620093 -1.622388 -0.3650137
    ## [6,] 0.5029143 0.06449492 1.2860242 1.0994436 -0.1121094 -1.813903 -0.3549308
    ##              D-O
    ## [1,]  0.27300455
    ## [2,]  0.24523475
    ## [3,]  0.01457709
    ## [4,]  0.02755159
    ## [5,]  0.48986409
    ## [6,] -0.67193397

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
    ## Acceptance Rate: 0.230735294117647

``` r
colnames(chain) <- c("intercept", "beta_lifeform", "beta_droid", "sigma")
head(chain)
```

    ##       intercept beta_lifeform beta_droid      sigma
    ## [1,] -0.3241930     0.8312493 -0.1999167 -0.2762768
    ## [2,] -0.7119803     1.1852332 -0.1618512 -0.4019959
    ## [3,] -1.9188958     2.8134818  1.8800293 -0.5020059
    ## [4,] -1.0688651     2.2115361  0.2356212 -0.4638475
    ## [5,] -1.0852703     2.5768966  0.6820284 -0.1239550
    ## [6,] -0.2834172     1.1843539 -0.7501749 -0.6964802

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

![](duration_data_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

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

![](duration_data_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

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

    ##                     2.5%        50%     97.5%
    ## intercept     -3.2330496 -0.1332490 3.0142606
    ## beta_lifeform -2.1908156  0.8082640 3.8685352
    ## beta_droid    -3.7260525 -0.6405153 2.4591854
    ## sigma         -0.9438854 -0.3921612 0.2158221

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
    ## 0.2002464 1.4162006 2.4697290

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
