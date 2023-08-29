#Atari Reinforcement Task (Results in Fig.6)

#Finetune
python3 main_rl.py --approach fine-tuning --seed 0 --date 220815

#EWC
python3 main_rl.py --approach ewc --reg_lambda 100000 --seed 0 --date 220815

#EWC+CPR
python3 main_rl.py --approach ewc_cpr --reg_lambda 100000 --cpr_lambda 0.05 --seed 0 --date 220815

#EWC+MCL
python3 main_rl.py --approach ewc_caf --reg_lambda 100000 --seed 0 --date 220815

#EWC+CAF
python3 main_rl.py --approach ewc_caf --reg_lambda 100000 --kld_lambda 0.01 --af_lambda 1e-5 --seed 0 --date 220815

#MAS
python3 main_rl.py --approach mas --reg_lambda 10 --seed 0 --date 220815

#MAS+CPR
python3 main_rl.py --approach mas_cpr --reg_lambda 10 --cpr_lambda 0.05 --seed 0 --date 220815

#MAS+MCL
python3 main_rl.py --approach mas_caf --reg_lambda 10 --seed 0 --date 220815

#MAS+CAF
python3 main_rl.py --approach mas_caf --reg_lambda 10 --kld_lambda 0.01 --af_lambda 1e-5 --seed 0 --date 220815

