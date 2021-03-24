library(tidyverse)

missing <- read_table('data/missing_probability.tsv') %>%
  mutate(above_critical = no > 2 * Ns) %>%
  mutate(jackknife = as.factor(j),
         Ns = as.factor(Ns))

ggplot(missing) + geom_point(aes(x=no, y=(prob), color=Ns, alpha = above_critical)) + 
  facet_grid(~ jackknife, labeller = label_both) +
  theme_minimal() +
  scale_color_brewer(palette = 'Set1')  +
  scale_y_log10() +
  guides(alpha = F) +
  labs(x = 'Sample size', y = 'Probability of missing a lineage') + 
  ggsave('fig/missing.pdf', width = 7, height = 4)

