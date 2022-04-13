Binary Edge Weight Model with INLA
================

This example covers fitting an edge weight model to presence/absence
(binary) data with an observation-level location effect.

*Note: Many of the procedures presented here are stochastic, and plots
and results may vary between compilations of this document. In
particular, MCMC chains and model estimates may sometimes not be
optimal.*

# Setup

First of all we’ll load in Rstan for model fitting in Stan, dplyr for
handling the data, and igraph for network plotting and computing network
centrality. We also load in a custom R file: “simulations.R” to generate
synthetic data for this example..

``` r
library(INLA)
library(dplyr)
library(igraph)

source("../scripts/simulations.R")
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
interpretation of edge weights. See the paper for further discussion on
this. `location` denotes the location at which the observation took
place, which may be relevant if location is likely to impact the
visibility of social events.

``` r
set.seed(123)
data <- simulate_binary()
df <- data$df
head(df)
```

    ##   node_1 node_2   type_1   type_2 event location
    ## 1    Rey   Leia Lifeform Lifeform     1        A
    ## 2    Rey   Leia Lifeform Lifeform     1        F
    ## 3    Rey   Leia Lifeform Lifeform     1        A
    ## 4    Rey   Leia Lifeform Lifeform     1        B
    ## 5    Rey   Leia Lifeform Lifeform     0        E
    ## 6    Rey   Leia Lifeform Lifeform     1        D

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
    ## # Groups:   node_1, node_2 [1]
    ##   node_1 node_2 type_1   type_2   event location dyad_id location_id
    ##   <fct>  <fct>  <fct>    <fct>    <int> <fct>      <int>       <int>
    ## 1 Rey    Leia   Lifeform Lifeform     1 A              1           1
    ## 2 Rey    Leia   Lifeform Lifeform     1 F              1           6
    ## 3 Rey    Leia   Lifeform Lifeform     1 A              1           1
    ## 4 Rey    Leia   Lifeform Lifeform     1 B              1           2
    ## 5 Rey    Leia   Lifeform Lifeform     0 E              1           5
    ## 6 Rey    Leia   Lifeform Lifeform     1 D              1           4

``` r
df_agg <- df %>%
  group_by(node_1, node_2) %>%
  summarise(event_count=sum(event), dyad_id=cur_group_id(), total_obs=n()) %>%
  mutate(node_1_id=as.integer(node_1), node_2_id=as.integer(node_2)) %>%
  mutate(sri=event_count/total_obs)

df_agg$node_1_type <- ""
df_agg$node_2_type <- ""
df_agg[df_agg$node_1_id <= 4, ]$node_1_type <- "l"
df_agg[df_agg$node_1_id >= 5, ]$node_1_type <- "d"
df_agg[df_agg$node_2_id <= 4, ]$node_2_type <- "l"
df_agg[df_agg$node_2_id >= 5, ]$node_2_type <- "d"
df_agg$dyad_type <- factor(paste0(df_agg$node_1_type, df_agg$node_2_type), levels=c("ll", "ld", "dd"))

num_obs <- nrow(df)
num_dyads <- nrow(df_agg)

head(df_agg)
```

    ## # A tibble: 6 × 11
    ## # Groups:   node_1 [1]
    ##   node_1 node_2  event_count dyad_id total_obs node_1_id node_2_id   sri
    ##   <fct>  <fct>         <int>   <int>     <int>     <int>     <int> <dbl>
    ## 1 Rey    Leia             29       1        31         1         2 0.935
    ## 2 Rey    Obi-Wan          14       2        14         1         3 1    
    ## 3 Rey    Luke             49       3        49         1         4 1    
    ## 4 Rey    C-3PO            23       4        41         1         5 0.561
    ## 5 Rey    BB-8             13       5        19         1         6 0.684
    ## 6 Rey    R2-D2            22       6        49         1         7 0.449
    ## # … with 3 more variables: node_1_type <chr>, node_2_type <chr>,
    ## #   dyad_type <fct>

# Fitting the model

To fit the model using INLA, we need to define it in the form of a GLM.
BISoN models don’t need an intercept (though they can be included in
modelling relative effects) so the intercept is excluded from the model
by adding a preceding `0` to the formula. Dyad ID is modelled (as a
factor) as a fixed effect, which will become our edge weight estimates,
and location is modelled as a random effect (also known as a varying
intercept). Because this model is for binary (presence/absence) data,
the GLM equivalent family is the binomial family. The model can be built
using the code below:

``` r
# Prepare dataframe by assigning factors
df$dyad_id <- as.factor(df$dyad_id)

# Set priors to match Stan model
prior.fixed <- list(mean=0, prec=1)
prior.random <- list(prec=list(prior="normal", param=c(0, 1)))

# Fit the INLA model
fit_edge <- inla(event ~ 0 + dyad_id + f(location, model="iid", hyper=prior.random), 
                 family="binomial", 
                 data=df,
                 control.fixed=prior.fixed,
                 control.compute=list(config = TRUE)
)
```

# Posterior predictive checks

To check the model has captured important aspects of the data well, we
can run a predictive check of the predictions of the model against the
observed event counts using the following code.

``` r
# Extract samples (including predictor values) from posterior of INLA model
inla_samples <- inla.posterior.sample(20, fit_edge)

# Plot the density of the observed event counts
plot(density(df_agg$event_count), main="", xlab="Dyadic event counts", ylim=c(0, 0.025))

# Plot the densities of the predicted event counts, repeat for multiple samples
df_copy <- df
for (i in 1:length(inla_samples)) {
  j <- sample(1:length(inla_samples), size=1)
  df_copy$event <- rbinom(nrow(df), 1, plogis(head(inla_samples[[j]]$latent, nrow(df))))
  df_agg_copy <- df_copy %>% 
    group_by(node_1, node_2) %>%
    summarise(event_count=sum(event))
  lines(density(df_agg_copy$event_count), col=rgb(0, 0, 1, 0.5))
}
```

![](ewm_binary_inla_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

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
# Extract posterior samples again, this time with enough samples to construct reliable CIs
num_samples <- 1000
inla_samples <- inla.posterior.sample(num_samples, fit_edge)
logit_edge_samples <- matrix(0, length(inla_samples), 28)
for (i in 1:length(inla_samples)) {
  logit_edge_samples[i, ] <- tail(inla_samples[[i]]$latent, 28)
}
edge_samples <- plogis(logit_edge_samples) # (0, 1) scale edge weights
```

We can summarise the distribution over edge lists by calculating the
credible intervals, indicating likely values for each edge. We’ll use
the 95% credible interval in this example, but there’s no reason to
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

    ##                   median  2.5% 97.5%
    ## Rey <-> Leia       0.866 0.724 0.949
    ## Rey <-> Obi-Wan    0.852 0.671 0.956
    ## Rey <-> Luke       0.935 0.844 0.978
    ## Rey <-> C-3PO      0.517 0.365 0.668
    ## Rey <-> BB-8       0.624 0.409 0.801
    ## Rey <-> R2-D2      0.436 0.295 0.580
    ## Rey <-> D-O        0.113 0.047 0.232
    ## Leia <-> Obi-Wan   0.925 0.829 0.976
    ## Leia <-> Luke      0.928 0.827 0.978
    ## Leia <-> C-3PO     0.215 0.092 0.398
    ## Leia <-> BB-8      0.878 0.755 0.950
    ## Leia <-> R2-D2     0.488 0.320 0.648
    ## Leia <-> D-O       0.127 0.050 0.247
    ## Obi-Wan <-> Luke   0.896 0.733 0.967
    ## Obi-Wan <-> C-3PO  0.592 0.345 0.811
    ## Obi-Wan <-> BB-8   0.886 0.714 0.965
    ## Obi-Wan <-> R2-D2  0.774 0.599 0.899
    ## Obi-Wan <-> D-O    0.245 0.138 0.380
    ## Luke <-> C-3PO     0.721 0.552 0.845
    ## Luke <-> BB-8      0.810 0.659 0.914
    ## Luke <-> R2-D2     0.354 0.213 0.509
    ## Luke <-> D-O       0.447 0.285 0.609
    ## C-3PO <-> BB-8     0.128 0.044 0.290
    ## C-3PO <-> R2-D2    0.067 0.023 0.164
    ## C-3PO <-> D-O      0.122 0.036 0.304
    ## BB-8 <-> R2-D2     0.148 0.050 0.340
    ## BB-8 <-> D-O       0.102 0.038 0.205
    ## R2-D2 <-> D-O      0.125 0.037 0.288

In social network analysis, a more useful format for network data is
usually adjacency matrices, rather than edge lists, so now we’ll convert
the distribution of edge lists to a distribution of adjacency matrices,
and store the result in an 8 x 8 x 4000 tensor, as there are 8 nodes and
4000 samples from the posterior.

``` r
adj_tensor <- array(0, c(8, 8, num_samples))
logit_adj_tensor <- array(0, c(8, 8, num_samples))
for (dyad_id in 1:num_dyads) {
  dyad_row <- df_agg[df_agg$dyad_id == dyad_id, ]
  adj_tensor[dyad_row$node_1_id, dyad_row$node_2_id, ] <- edge_samples[, dyad_id]
  logit_adj_tensor[dyad_row$node_1_id, dyad_row$node_2_id, ] <- logit_edge_samples[, dyad_id]
}
adj_tensor[, , 1] # Print the first sample of the posterior distribution over adjacency matrices
```

    ##      [,1]      [,2]      [,3]      [,4]      [,5]      [,6]      [,7]
    ## [1,]    0 0.7957468 0.8341600 0.9092669 0.4950734 0.5399023 0.4070415
    ## [2,]    0 0.0000000 0.9606458 0.9115291 0.1304114 0.7701484 0.5196656
    ## [3,]    0 0.0000000 0.0000000 0.9074307 0.5980924 0.8551566 0.7890884
    ## [4,]    0 0.0000000 0.0000000 0.0000000 0.6658502 0.8489075 0.3918974
    ## [5,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.1477538 0.1170949
    ## [6,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.1172482
    ## [7,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000
    ## [8,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000
    ##            [,8]
    ## [1,] 0.16620518
    ## [2,] 0.08694004
    ## [3,] 0.35271912
    ## [4,] 0.36315155
    ## [5,] 0.28027757
    ## [6,] 0.21963793
    ## [7,] 0.11459535
    ## [8,] 0.00000000

The adjacency matrix above corresponds to a single draw of the posterior
adjacency matrices. You’ll notice the edges have been transformed back
to the \[0, 1\] range from the logit scale using the logistic function.
If there are no additional effects (such as location in our case), the
transformed edge weights will be probabilities and the median will be
approximately the same as the simple ratio index for each dyad. However,
when additional effects are included, the transformed values can no
longer be interpreted as probabilities, though they will be useful for
visualisation and analysis purposes.

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
adj_quantiles <- apply(adj_tensor, c(1, 2), function(x) quantile(x, probs=c(0.025, 0.5, 0.975)))
adj_lower <- adj_quantiles[1, , ]
adj_mid <- adj_quantiles[2, , ]
adj_upper <- adj_quantiles[3, , ]

# Calculate width of credible intervals.
adj_range <- adj_upper - adj_lower
adj_range[is.nan(adj_range)] <- 0

# Generate two igraph objects, one form the median and one from the standardised width.
g_mid <- graph_from_adjacency_matrix(adj_mid, mode="undirected", weighted=TRUE)
g_range <- graph_from_adjacency_matrix(adj_range, mode="undirected", weighted=TRUE)

# Plot the median graph first and then the standardised width graph to show uncertainty over edges.
coords <- igraph::layout_nicely(g_mid)
plot(g_mid, edge.width=3 * E(g_mid)$weight, edge.color="black",  layout=coords)
plot(g_mid, edge.width=20 * E(g_range)$weight, edge.color=rgb(0, 0, 0, 0.25), 
     vertex.label=c("Rey", "Leia", "Obi-Wan", "Luke", "C-3PO", "BB-8", "R2-D2", "D-O"), 
     vertex.label.dist=4, vertex.label.color="black", layout=coords, add=TRUE)
```

![](ewm_binary_inla_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

This plot can be extended in multiple ways, for example by thresholding
low edge weights to visualise the network more tidily, or by adding
halos around nodes to show uncertainty around node centrality, and so
on.

# Next Steps

Now the edge weight model has been fitted, the edge weight posteriors
can be used in the various types of network analyses shown in this
repository.

Save the edge weights for further analysis.

``` r
data_inla=list(df=df, df_agg=df_agg, logit_edge_samples=logit_edge_samples)
saveRDS(data_inla, file="../example_data/binary_inla.RData")
```

# Compare to Stan model

Out of curiosity, and as a basic check that the models are performing
equivalently, check the edge weight estimates from the INLA model
against the Stan model:

``` r
data <- readRDS("../example_data/binary.RData")
data_inla <- readRDS("../example_data/binary_inla.RData")
logit_edge_samples <- data$logit_edge_samples
logit_edge_samples_inla <- data_inla$logit_edge_samples

plot(logit_edge_samples[1, ], logit_edge_samples_inla[1, ])
for (i in 2:21) {
  points(logit_edge_samples[i, ], logit_edge_samples_inla[i, ])
}
abline(a=0, b=1)
```

![](ewm_binary_inla_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->
