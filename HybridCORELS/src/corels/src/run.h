#ifndef RUN_H
#define RUN_H

#include "rule.h"
#include "queue.h"

int run_corels_begin(double c, char* vstring, int curiosity_policy,
                  int map_type, int ablation, int calculate_size, int nrules, int nlabels,
                  int nsamples, rule_t* rules, rule_t* labels, rule_t* bb_errors, rule_t* meta, int freq, char* log_fname,
                  PermutationMap*& pmap, CacheTree*& tree, Queue*& queue, double& init,
                  std::set<std::string>& verbosity, double beta, double min_coverage, double max_coverage_disparity,
                  rule_t* sensitive_groups, int n_sensitive_groups, int* inconsistent_groups_indices_c,
                  int* inconsistent_groups_min_card_c, int* inconsistent_groups_max_card_c, int nb_incons_groups_c);

int run_corels_loop(size_t max_num_nodes, PermutationMap* pmap, CacheTree* tree, Queue* queue);

double run_corels_end(std::vector<int>* rulelist, std::vector<int>* classes, int early, int latex_out, rule_t* rules,
                      rule_t* labels, char* opt_fname, PermutationMap*& pmap, CacheTree*& tree, Queue*& queue,
                      double init, std::set<std::string>& verbosity, std::vector<int>* rules_support, std::vector<double>* rules_accuracy);

#endif
