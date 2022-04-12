Count Data Example
================

This example covers fitting a edge weight model to count data (where the
count of social events per observation is recorded) with an
observation-level location effect, basic model checking and diagnostics,
visualising networks with uncertainty, calculating probability
distributions over network centrality, and propagating network
uncertainty into subsequent analyses.

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

Now we will simulate data using the `simulate_count()` function. The
rows of the resulting dataframe describe observations at the dyadic
level between nodes. In this dataframe, `event_count` denotes the number
of social events observed in each observation period. The exact
definition of observation period will depend on the study, but is
commonly a sampling period where at least one of the members of the dyad
was observed. This can also be a sampling period where both members of
the dyad were observed, and the distinction will affect the
interpretation of edge weights. See the paper for further discussion on
this. `location` denotes the location at which the observation took
place, which may be relevant if location is likely to impact the
visibility of social events.

``` r
set.seed(1)
data <- simulate_count()
df <- data$df
head(df)
```

    ##   node_1 node_2   type_1   type_2 event_count duration location
    ## 1    Rey   Leia Lifeform Lifeform           1        3        A
    ## 2    Rey   Leia Lifeform Lifeform           1        5        F
    ## 3    Rey   Leia Lifeform Lifeform           0        5        F
    ## 4    Rey   Leia Lifeform Lifeform           1        8        F
    ## 5    Rey   Leia Lifeform Lifeform           6        9        D
    ## 6    Rey   Leia Lifeform Lifeform           3        4        D

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
  mutate(location_id=as.integer(location), duration=as.integer(duration))
head(df)
```

    ## # A tibble: 6 × 9
    ## # Groups:   node_1, node_2 [1]
    ##   node_1 node_2 type_1  type_2 event_count duration location dyad_id location_id
    ##   <fct>  <fct>  <fct>   <fct>        <int>    <int> <fct>      <int>       <int>
    ## 1 Rey    Leia   Lifefo… Lifef…           1        3 A              1           1
    ## 2 Rey    Leia   Lifefo… Lifef…           1        5 F              1           6
    ## 3 Rey    Leia   Lifefo… Lifef…           0        5 F              1           6
    ## 4 Rey    Leia   Lifefo… Lifef…           1        8 F              1           6
    ## 5 Rey    Leia   Lifefo… Lifef…           6        9 D              1           4
    ## 6 Rey    Leia   Lifefo… Lifef…           3        4 D              1           4

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
    ## 1 Rey    Leia                   75       1         1         2
    ## 2 Rey    Obi-Wan                14       2         1         3
    ## 3 Rey    Luke                  367       3         1         4
    ## 4 Rey    C-3PO                   7       4         1         5
    ## 5 Rey    BB-8                    2       5         1         6
    ## 6 Rey    R2-D2                   2       6         1         7

Now we have all of the data in the right format for fitting the model,
we just need to put it into a list object. The data required by the
statistical model is defined in `count_model.stan`.

``` r
model_data <- list(
  num_obs=nrow(df), # Number of observations
  num_dyads=nrow(df_agg), # Number of dyads
  num_locations=6, # Number of locations
  dyad_ids=df$dyad_id, # Vector of dyad IDs corresponding to each observation
  location_ids=df$location_id, # Vector of location IDs corresponding to each observation
  event_count=df$event_count, # Vector of event counts corresponding to each observation,
  durations=df$duration
)
```

# Fitting the model

To fit the model, we first must compile it and load it into memory using
the function `stan_model()` and providing the filepath to the model. The
working directory will need to be set to the directory of the model for
this to work properly.

``` r
edge_model <- stan_model("../models/edge_count.stan")
```

Compiling the model may take a minute or two, but once this is done, the
model can be fit using `sampling()`. The argument `cores` sets the
number of CPU cores to be used for fitting the model, if your computer
has 4 or more cores, it’s worth setting this to 4.

``` r
fit_edge <- sampling(edge_model, model_data, cores=4)
```

    ## Warning: Bulk Effective Samples Size (ESS) is too low, indicating posterior means and medians may be unreliable.
    ## Running the chains for more iterations may help. See
    ## https://mc-stan.org/misc/warnings.html#bulk-ess

# Model checking

The R-hat values provided by Stan indicate how well the chains have
converged, with values very close to 1.00 being ideal. Values diverging
from 1.00 indicate that the posterior samples may be very unreliable,
and shouldn’t be trusted. The chains can be plotted using Rstan’s
`traceplot` function to verify this visually:

``` r
traceplot(fit_edge)
```

![](ewm_count_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

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
event_pred <- extract(fit_edge)$event_pred
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

![](ewm_count_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

This plot shows that the observed data falls well within the predicted
densities, and the predictions suggest the model has captured the main
features of the data well. Now we can be reasonably confident that the
model has fit correctly and describes the data well, so we can start to
make inferences from the model.

# Extracting edge weights

The main purpose of this part of the framework is to estimate edge
weights of dyads. We can access these using the `logit_p` quantity. This
will give a distribution of logit-scale edge weights for each dyad, akin
to an edge list. We’ll apply the logistic function `plogis` to get the
edge weights back to their original scale:

``` r
log_edge_samples <- extract(fit_edge)$log_edge # Logit scale edge weights
edge_samples <- exp(log_edge_samples) # (0, 1) scale edge weights
```

We can summarise the distribution over edge lists by calculating the
credible intervals, indicating likely values for each edge. We’ll use
the 89% credible interval in this example, but there’s no reason to
choose this interval over any other. The distribution over edge lists
can be summarised in the following code:

``` r
dyad_name <- do.call(paste, c(df_agg[c("node_1", "node_2")], sep=" <-> "))
edge_lower <- apply(edge_samples, 2, function(x) quantile(x, probs=0.025))
edge_upper <- apply(edge_samples, 2, function(x) quantile(x, probs=0.975))
edge_median <- apply(edge_samples, 2, function(x) quantile(x, probs=0.5))
edge_list <- cbind(
  "median"=round(edge_median, 3), 
  "2.5%"=round(edge_lower, 3), 
  "97.5%"=round(edge_upper, 3)
)
rownames(edge_list) <- dyad_name
edge_list
```

    ##                   median   2.5%  97.5%
    ## Rey <-> Leia       4.576  2.941  7.272
    ## Rey <-> Obi-Wan    0.741  0.396  1.345
    ## Rey <-> Luke      53.511 36.037 81.414
    ## Rey <-> C-3PO      0.655  0.294  1.350
    ## Rey <-> BB-8       0.158  0.060  0.361
    ## Rey <-> R2-D2      0.291  0.109  0.720
    ## Rey <-> D-O        2.450  1.375  4.307
    ## Leia <-> Obi-Wan   1.846  1.132  3.089
    ## Leia <-> Luke     22.721 15.286 34.274
    ## Leia <-> C-3PO     2.112  1.288  3.459
    ## Leia <-> BB-8      0.639  0.308  1.243
    ## Leia <-> R2-D2     2.028  1.175  3.401
    ## Leia <-> D-O       3.625  2.379  5.673
    ## Obi-Wan <-> Luke  17.696 11.733 27.581
    ## Obi-Wan <-> C-3PO  0.513  0.227  1.048
    ## Obi-Wan <-> BB-8   0.375  0.101  1.111
    ## Obi-Wan <-> R2-D2  0.591  0.273  1.176
    ## Obi-Wan <-> D-O    1.197  0.329  3.630
    ## Luke <-> C-3PO     3.154  1.953  5.148
    ## Luke <-> BB-8      0.587  0.258  1.191
    ## Luke <-> R2-D2     2.914  1.678  4.985
    ## Luke <-> D-O       9.543  6.335 14.612
    ## C-3PO <-> BB-8    10.558  7.066 16.270
    ## C-3PO <-> R2-D2    9.034  6.026 13.794
    ## C-3PO <-> D-O     15.771 10.461 24.291
    ## BB-8 <-> R2-D2     2.453  1.504  4.088
    ## BB-8 <-> D-O       1.539  0.694  3.266
    ## R2-D2 <-> D-O      8.440  4.881 15.093

In social network analysis, a more useful format for network data is
usually adjacency matrices, rather than edge lists, so now we’ll convert
the distribution of edge lists to a distribution of adjacency matrices,
and store the result in an 8 x 8 x 4000 tensor, as there are 8 nodes and
4000 samples from the posterior.

``` r
adj_tensor <- array(0, c(8, 8, num_iterations))
log_adj_tensor <- array(0, c(8, 8, num_iterations))
for (dyad_id in 1:model_data$num_dyads) {
  dyad_row <- df_agg[df_agg$dyad_id == dyad_id, ]
  adj_tensor[dyad_row$node_1_id, dyad_row$node_2_id, ] <- edge_samples[, dyad_id]
  log_adj_tensor[dyad_row$node_1_id, dyad_row$node_2_id, ] <- log_edge_samples[, dyad_id]
}
adj_tensor[, , 1] # Print the first sample of the posterior distribution over adjacency matrices
```

    ##      [,1]     [,2]      [,3]     [,4]      [,5]       [,6]       [,7]      [,8]
    ## [1,]    0 6.109596 0.9872705 69.82662 1.2944544  0.2052630  0.3935842  2.828884
    ## [2,]    0 0.000000 1.8456417 26.00284 2.8700301  0.5826202  3.0678275  3.875306
    ## [3,]    0 0.000000 0.0000000 21.81286 0.9835756  0.1321086  0.5048463  1.476033
    ## [4,]    0 0.000000 0.0000000  0.00000 5.2907046  0.3310766  3.2187944 12.811757
    ## [5,]    0 0.000000 0.0000000  0.00000 0.0000000 13.8986763 12.3799810 18.915245
    ## [6,]    0 0.000000 0.0000000  0.00000 0.0000000  0.0000000  3.7821415  2.617301
    ## [7,]    0 0.000000 0.0000000  0.00000 0.0000000  0.0000000  0.0000000 10.508799
    ## [8,]    0 0.000000 0.0000000  0.00000 0.0000000  0.0000000  0.0000000  0.000000

The adjacency matrix above corresponds to a single draw of the posterior
adjacency matrices. You’ll notice the edges have been transformed back
to the \>0 range from the log-scale scale using the exponential
function. If there are no additional effects (such as location in our
case), the transformed edge weights will be probabilities and the median
will be approximately the same as the simple ratio index for each dyad.
However, when additional effects are included, the transformed values
can no longer be interpreted directly as rates, though they will be
useful for visualisation and analysis purposes.

# Visualising uncertainty

The aim of our network visualisation is to plot a network where the
certainty in edge weights (edge weights) can be seen. To do this we’ll
use a semi-transparent line around each edge with a width that
corresponds to a uncertainty measures. The uncertainty measure will
simply be the difference between the 97.5% and 2.5% credible interval
estimate for each edge weight. We can calculate this from the
transformed adjacency tensor object, generate two igraph objects for the
main network and the uncertainty in edges, and plot them with the same
coordinates.

``` r
# Calculate lower, median, and upper quantiles of edge weights. Lower and upper give credible intervals.
adj_quantiles <- apply(adj_tensor, c(1, 2), function(x) quantile(x, probs=c(0.025, 0.5, 0.975)))
adj_lower <- adj_quantiles[1, , ]
adj_mid <- adj_quantiles[2, , ]
adj_upper <- adj_quantiles[3, , ]

# Calculate width/range of credible intervals.
adj_range <- ((adj_upper - adj_lower))
adj_range[is.nan(adj_range)] <- 0

# Generate two igraph objects, one form the median and one from the width.
g_mid <- graph_from_adjacency_matrix(adj_mid, mode="undirected", weighted=TRUE)
g_range <- graph_from_adjacency_matrix(adj_range, mode="undirected", weighted=TRUE)

# Plot the median graph first and then the width graph to show uncertainty over edges.
coords <- igraph::layout_nicely(g_mid)
plot(g_mid, edge.width=0.1 * E(g_mid)$weight, edge.color="black",  layout=coords)
plot(g_mid, edge.width=E(g_range)$weight, edge.color=rgb(0, 0, 0, 0.25), 
     vertex.label=c("Rey", "Leia", "Obi-Wan", "Luke", "C-3PO", "BB-8", "R2-D2", "D-O"), 
     vertex.label.dist=4, vertex.label.color="black", layout=coords, add=TRUE)
```

![](ewm_count_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

This plot can be extended in multiple ways, for example by thresholding
low edge weights to visualise the network more tidily, or by adding
halos around nodes to show uncertainty around network centrality, and so
on.

# Next Steps

Now the edge weight model has been fitted, the edge weight posteriors
can be used in the various types of network analyses shown in this
repository.
