#!/bin/bash

# Get all dense network results
cd dense_network
python dense_out_part.py
python dense_phase.py

# Get all otuput-split network results
cd ../split_network
python split_out_part.py
python split_phase.py

# Get shallow network results
cd ../shallow_network
python shallow_out_part.py
cd ..

# Plot non-systematic norm vs generations
cd norm_vs_gen
cd dense_network
python dense_out_part.py
#python dense_phase.py
cp dense_norm_v_gen.txt ..
cd ../split_network
python split_out_part.py
#python split_phase.py
cp split_norm_v_gen.txt ..
cd ../shallow_network
python shallow_out_part.py
cp shallow_norm_v_gen.txt ..
cd ..
python merge_refine.py
cd ..

# Plot other learning rules dynamics
#cd other_rules/anti_hebbian
#python dense_out_hebb.py
#cd ../contrastive_hebbian
#python dense_out_hebb.py
#cd ../hebbian
#python dense_out_hebb.py
#cd ../predictive_coding
#python dense_out_hebb.py
#cd ../..
