#!/bin/bash

rm -rf {dense_network,shallow_network,split_network,norm_vs_gen,other_rules}/{*.pdf,*.png,*.txt}
cd norm_vs_gen/
rm -rf {dense_network,shallow_network,split_network}/{*.pdf,*.png,*.txt}
cd ../other_rules/
rm -rf {anti_hebbian,contrastive_hebbian,hebbian,predictive_coding}/{*.pdf,*.png,*.txt}
cd ..
