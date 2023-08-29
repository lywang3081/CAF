
#S-CIFAR-100 (Results in Fig.2, Fig.5, Fig.C1, Fig.C2, Fig.C3, Fig.C4, Fig.C5)

#EWC
python3 ./main.py --experiment s_cifar100 --approach ewc --lamb 10000 --seed 0

#EWC+CPR
python3 ./main.py --experiment s_cifar100 --approach ewc --lamb 10000 --lamb_cpr 1.5 --seed 0

#EWC+AF-1
python3 ./main.py --experiment s_cifar100 --approach ewc --lamb 10000 --lamb_af 1e-6 --seed 0

#EWC+AF-2
python3 ./main.py --experiment s_cifar100 --approach ewc_af2 --lamb 10000 --lamb_af 1 --seed 0

#EWC+MCL(low diversity)
python3 ./main.py --experiment s_cifar100 --approach ewc_caf --mcl mcl-l --lamb 10000 --seed 0

#EWC+MCL(low diversity, original width)
python3 ./main.py --experiment s_cifar100 --approach ewc_caf --mcl mcl-l --original_width --lamb 10000 --seed 0

#EWC+MCL(medium diversity)
python3 ./main.py --experiment s_cifar100 --approach ewc_caf --mcl mcl-m --lamb 10000 --seed 0

#EWC+MCL(medium diversity, original width)
python3 ./main.py --experiment s_cifar100 --approach ewc_caf --mcl mcl-m --original_width --lamb 10000 --seed 0

#EWC+MCL(high diversity)
python3 ./main.py --experiment s_cifar100 --approach ewc_caf --mcl mcl-h --lamb 10000 --seed 0

#EWC+MCL(high diversity, original width)
python3 ./main.py --experiment s_cifar100 --approach ewc_caf --mcl mcl-h --original_width --lamb 10000 --seed 0

#EWC+CAF(high diversity)
python3 ./main.py --experiment s_cifar100 --approach ewc_caf --mcl mcl-h --lamb 10000 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#EWC+CAF(high diversity, original width)
python3 ./main.py --experiment s_cifar100 --approach ewc_caf --mcl mcl-h --original_width --lamb 10000 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#MAS
python3 ./main.py --experiment s_cifar100 --approach mas --lamb 16 --seed 0

#MAS+CPR
python3 ./main.py --experiment s_cifar100 --approach mas --lamb 16 --lamb_cpr 1.5 --seed 0

#MAS+AF-1
python3 ./main.py --experiment s_cifar100 --approach mas --lamb 16 --lamb_af 1e-6 --seed 0

#MAS+AF-2
python3 ./main.py --experiment s_cifar100 --approach mas_af2 --lamb 16 --lamb_emp 1 --seed 0

#MAS+MCL(low diversity)
python3 ./main.py --experiment s_cifar100 --approach mas_caf --mcl mcl-l --lamb 16 --seed 0

#MAS+MCL(low diversity, original width)
python3 ./main.py --experiment s_cifar100 --approach mas_caf --mcl mcl-l  --original_width --lamb 16 --seed 0

#MAS+MCL(medium diversity)
python3 ./main.py --experiment s_cifar100 --approach mas_caf --mcl mcl-m --lamb 16 --seed 0

#MAS+MCL(medium diversity, original width)
python3 ./main.py --experiment s_cifar100 --approach mas_caf --mcl mcl-m  --original_width --lamb 16 --seed 0

#MAS+MCL(high diversity)
python3 ./main.py --experiment s_cifar100 --approach mas_caf --mcl mcl-h --lamb 16 --seed 0

#MAS+MCL(high diversity, original width)
python3 ./main.py --experiment s_cifar100 --approach mas_caf --mcl mcl-h  --original_width --lamb 16 --seed 0

#MAS+CAF(high diversity)
python3 ./main.py --experiment s_cifar100 --approach mas_caf --mcl mcl-h --lamb 16 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#MAS+CAF(high diversity, original width)
python3 ./main.py --experiment s_cifar100 --approach mas_caf --mcl mcl-h  --original_width --lamb 16 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#RWALK
python3 ./main.py --experiment s_cifar100 --approach rwalk --lamb 128 --seed 0

#SI
python3 ./main.py --experiment s_cifar100 --approach si --lamb 8 --seed 0

#AGS-CL
python3 ./main.py --experiment s_cifar100 --approach gs --lamb 3200 --mu 10 --rho 0.3 --eta 0.9 --seed 0

#Experience Replay(ER)
python3 ./main.py --experiment s_cifar100 --approach er --seed 0

#Finetune
python3 ./main.py --experiment s_cifar100 --approach ft --seed 0



#R-CIFAR-100 (Results in Fig.2, Fig.5, Fig.C1, Fig.C2, Fig.C4, Fig.C5)

#EWC
python3 ./main.py --experiment r_cifar100 --approach ewc --lamb 10000 --seed 0

#EWC+CPR
python3 ./main.py --experiment r_cifar100 --approach ewc --lamb 10000 --lamb_cpr 1.5 --seed 0

#EWC+AF-1
python3 ./main.py --experiment r_cifar100 --approach ewc --lamb 10000 --lamb_af 1e-6 --seed 0

#EWC+AF-2
python3 ./main.py --experiment r_cifar100 --approach ewc_af2 --lamb 10000 --lamb_af 1 --seed 0

#EWC+MCL(low diversity)
python3 ./main.py --experiment r_cifar100 --approach ewc_caf --mcl mcl-l --lamb 10000 --seed 0

#EWC+MCL(low diversity, original width)
python3 ./main.py --experiment r_cifar100 --approach ewc_caf --mcl mcl-l --original_width --lamb 10000 --seed 0

#EWC+MCL(medium diversity)
python3 ./main.py --experiment r_cifar100 --approach ewc_caf --mcl mcl-m --lamb 10000 --seed 0

#EWC+MCL(medium diversity, original width)
python3 ./main.py --experiment r_cifar100 --approach ewc_caf --mcl mcl-m --original_width --lamb 10000 --seed 0

#EWC+MCL(high diversity)
python3 ./main.py --experiment r_cifar100 --approach ewc_caf --mcl mcl-h --lamb 10000 --seed 0

#EWC+MCL(high diversity, original width)
python3 ./main.py --experiment r_cifar100 --approach ewc_caf --mcl mcl-h --original_width --lamb 10000 --seed 0

#EWC+CAF(high diversity)
python3 ./main.py --experiment r_cifar100 --approach ewc_caf --mcl mcl-h --lamb 10000 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#EWC+CAF(high diversity, original width)
python3 ./main.py --experiment r_cifar100 --approach ewc_caf --mcl mcl-h --original_width --lamb 10000 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#MAS
python3 ./main.py --experiment r_cifar100 --approach mas --lamb 4 --seed 0

#MAS+CPR
python3 ./main.py --experiment r_cifar100 --approach mas --lamb 4 --lamb_cpr 1.5 --seed 0

#MAS+AF-1
python3 ./main.py --experiment r_cifar100 --approach mas --lamb 4 --lamb_af 1e-6 --seed 0

#MAS+AF-2
python3 ./main.py --experiment r_cifar100 --approach mas_af2 --lamb 4 --lamb_emp 1 --seed 0

#MAS+MCL(low diversity)
python3 ./main.py --experiment r_cifar100 --approach mas_caf --mcl mcl-l --lamb 4 --seed 0

#MAS+MCL(low diversity, original width)
python3 ./main.py --experiment r_cifar100 --approach mas_caf --mcl mcl-l --original_width --lamb 4 --seed 0

#MAS+MCL(medium diversity)
python3 ./main.py --experiment r_cifar100 --approach mas_caf --mcl mcl-m --lamb 4 --seed 0

#MAS+MCL(medium diversity, original width)
python3 ./main.py --experiment r_cifar100 --approach mas_caf --mcl mcl-m --original_width --lamb 4 --seed 0

#MAS+MCL(high diversity)
python3 ./main.py --experiment r_cifar100 --approach mas_caf --mcl mcl-h --lamb 4 --seed 0

#MAS+MCL(high diversity, original width)
python3 ./main.py --experiment r_cifar100 --approach mas_caf --mcl mcl-h --original_width --lamb 4 --seed 0

#MAS+CAF(high diversity)
python3 ./main.py --experiment r_cifar100 --approach mas_caf --mcl mcl-h --lamb 4 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#MAS+CAF(high diversity, original width)
python3 ./main.py --experiment r_cifar100 --approach mas_caf --mcl mcl-h --original_width --lamb 4 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#RWALK
python3 ./main.py --experiment r_cifar100 --approach rwalk --lamb 6 --seed 0

#SI
python3 ./main.py --experiment r_cifar100 --approach si --lamb 10 --seed 0

#AGS-CL
python3 ./main.py --experiment r_cifar100 --approach gs --lamb 1600 --mu 10 --rho 0.3 --eta 0.9 --seed 0

#Experience Replay(ER)
python3 ./main.py --experiment r_cifar100 --approach er --seed 0

#Finetune
python3 ./main.py --experiment r_cifar100 --approach ft --seed 0



#R-CIFAR-10/100 (Results in Fig.2, Fig.5, Fig.C1, Fig.C2, Fig.C3, Fig.C4, Fig.C5)

#EWC
python3 ./main.py --experiment r_cifar10_100 --approach ewc --lamb 20000 --seed 0

#EWC+CPR
python3 ./main.py --experiment r_cifar10_100 --approach ewc --lamb 20000 --lamb_cpr 0.5 --seed 0

#EWC+AF-1
python3 ./main.py --experiment r_cifar10_100 --approach ewc --lamb 20000 --lamb_af 1e-6 --seed 0

#EWC+AF-2
python3 ./main.py --experiment r_cifar10_100 --approach ewc_af2 --lamb 20000 --lamb_af 1 --seed 0

#EWC+MCL(low diversity)
python3 ./main.py --experiment r_cifar10_100 --approach ewc_caf --mcl mcl-l --lamb 20000 --seed 0

#EWC+MCL(low diversity, original width)
python3 ./main.py --experiment r_cifar10_100 --approach ewc_caf --mcl mcl-l --original_width --lamb 20000 --seed 0

#EWC+MCL(medium diversity)
python3 ./main.py --experiment r_cifar10_100 --approach ewc_caf --mcl mcl-m --lamb 20000 --seed 0

#EWC+MCL(medium diversity, original width)
python3 ./main.py --experiment r_cifar10_100 --approach ewc_caf --mcl mcl-m --original_width --lamb 20000 --seed 0

#EWC+MCL(high diversity)
python3 ./main.py --experiment r_cifar10_100 --approach ewc_caf --mcl mcl-h --lamb 20000 --seed 0

#EWC+MCL(high diversity, original width)
python3 ./main.py --experiment r_cifar10_100 --approach ewc_caf --mcl mcl-h --original_width --lamb 20000 --seed 0

#EWC+CAF(high diversity)
python3 ./main.py --experiment r_cifar10_100 --approach ewc_caf --mcl mcl-h --lamb 20000 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#EWC+CAF(high diversity, original width)
python3 ./main.py --experiment r_cifar10_100 --approach ewc_caf --mcl mcl-h --original_width --lamb 20000 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#MAS
python3 ./main.py --experiment r_cifar10_100 --approach mas --lamb 8 --seed 0

#MAS+CPR
python3 ./main.py --experiment r_cifar10_100 --approach mas --lamb 8 --lamb_cpr 0.2 --seed 0

#MAS+AF-1
python3 ./main.py --experiment r_cifar10_100 --approach mas --lamb 8 --lamb_af 1e-6 --seed 0

#MAS+AF-2
python3 ./main.py --experiment r_cifar10_100 --approach mas_af2 --lamb 8 --lamb_emp 1 --seed 0

#MAS+MCL(low diversity)
python3 ./main.py --experiment r_cifar10_100 --approach mas_caf --mcl mcl-l --lamb 8 --seed 0

#MAS+MCL(low diversity, original width)
python3 ./main.py --experiment r_cifar10_100 --approach mas_caf --mcl mcl-l --original_width --lamb 8 --seed 0

#MAS+MCL(medium diversity)
python3 ./main.py --experiment r_cifar10_100 --approach mas_caf --mcl mcl-m --lamb 8 --seed 0

#MAS+MCL(medium diversity, original width)
python3 ./main.py --experiment r_cifar10_100 --approach mas_caf --mcl mcl-m --original_width --lamb 8 --seed 0

#MAS+MCL(high diversity)
python3 ./main.py --experiment r_cifar10_100 --approach mas_caf --mcl mcl-h --lamb 8 --seed 0

#MAS+MCL(high diversity, original width)
python3 ./main.py --experiment r_cifar10_100 --approach mas_caf --mcl mcl-h --original_width --lamb 8 --seed 0

#MAS+CAF(high diversity)
python3 ./main.py --experiment r_cifar10_100 --approach mas_caf --mcl mcl-h --lamb 8 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#MAS+CAF(high diversity, original width)
python3 ./main.py --experiment r_cifar10_100 --approach mas_caf --mcl mcl-h --original_width --lamb 8 --lamb_kld 0.02 --lamb_af 1e-8 --seed 0

#RWALK
python3 ./main.py --experiment r_cifar10_100 --approach rwalk --lamb 40 --seed 0

#SI
python3 ./main.py --experiment r_cifar10_100 --approach si --lamb 6 --seed 0

#AGS-CL
python3 ./main.py --experiment r_cifar10_100 --approach gs --lamb 7000 --mu 20 --rho 0.2 --eta 0.9 --seed 0

#Experience Replay(ER)
python3 ./main.py --experiment r_cifar10_100 --approach er --seed 0

#Finetune
python3 ./main.py --experiment r_cifar10_100 --approach ft --seed 0



#Omniglot (Results in Fig.5)

#EWC
python3 ./main.py --experiment omniglot --approach ewc --lamb 5000000 --seed 0

#EWC+CPR
python3 ./main.py --experiment omniglot --approach ewc --lamb 5000000 --lamb_cpr 0.5 --seed 0

#EWC+CAF
python3 ./main.py --experiment omniglot --approach ewc_caf --mcl mcl-h --lamb 5000000 --lamb_kld 0.05 --lamb_af 1e-12 --seed 0

#MAS
python3 ./main.py --experiment omniglot --approach mas --lamb 4 --seed 0

#MAS+CPR
python3 ./main.py --experiment omniglot --approach mas --lamb 4 --lamb_cpr 0.2 --seed 0

#MAS+CAF
python3 ./main.py --experiment omniglot --approach mas_caf --mcl mcl-h --lamb 30 --lamb_kld 0.05 --lamb_af 1e-12 --seed 0

#RWALK
python3 ./main.py --experiment omniglot --approach rwalk --lamb 70 --seed 0

#SI
python3 ./main.py --experiment omniglot --approach si --lamb 10 --seed 0

#AGS-CL
python3 ./main.py --experiment omniglot --approach gs --lamb 1000 --mu 7 --rho 0.5 --eta 0.9 --seed 0

#Experience Replay(ER)
python3 ./main.py --experiment omniglot --approach er --seed 0

#Finetune
python3 ./main.py --experiment omniglot --approach ft --seed 0



#CUB-200-2011 (Results in Fig.5)

#EWC
python3 ./main.py --dataset CUB200 --trainer ewc --lamb 5 --tasknum 10 --seed 0

#EWC+CPR
python3 ./main.py --dataset CUB200 --trainer ewc --lamb 5 --lamb_cpr 1e-3 --tasknum 10 --seed 0

#EWC+CAF
python3 ./main.py --dataset CUB200 --trainer ewc_caf --lamb 5 --lamb_kld 1e-3 --lamb_af 1e-7 --tasknum 10 --seed 0

#MAS
python3 ./main.py --dataset CUB200 --trainer mas --lamb 0.05 --tasknum 10 --seed 0

#MAS+CPR
python3 ./main.py --dataset CUB200 --trainer mas --lamb 0.05 --lamb_cpr 1e-3 --tasknum 10 --seed 0

#MAS+CAF
python3 ./main.py --dataset CUB200 --trainer mas_caf --lamb 0.05 --lamb_kld 1e-3 --lamb_af 1e-7 --tasknum 10 --seed 0

#RWALK
python3 ./main.py --dataset CUB200 --trainer rwalk --lamb 25 --tasknum 10 --seed 0

#SI
python3 ./main.py --dataset CUB200 --trainer si --lamb 0.4 --tasknum 10 --seed 0

#AGS-CL
python3 ./main.py --experiment CUB200 --approach gs --lamb 1e-4 --mu 1e-3 --rho 0.5 --eta 0.9 --tasknum 10 --seed 0



#Tiny-ImageNet (Results in Fig.5)

#EWC
python3 ./main.py --dataset tinyImageNet --trainer ewc --lamb 40 --tasknum 10 --seed 0

#EWC+CPR
python3 ./main.py --dataset tinyImageNet --trainer ewc --lamb 40 --lamb_cpr 0.6 --tasknum 10 --seed 0

#EWC+CAF
python3 ./main.py --dataset tinyImageNet --trainer ewc_caf --lamb 320 --lamb_kld 1e-3 --lamb_af 1e-7 --tasknum 10 --seed 0

#MAS
python3 ./main.py --dataset tinyImageNet --trainer mas --lamb 0.5 --tasknum 10 --seed 0

#MAS+CPR
python3 ./main.py --dataset tinyImageNet --trainer mas --lamb 0.5 --lamb_cpr 0.01 --tasknum 10 --seed 0

#MAS+CAF
python3 ./main.py --dataset tinyImageNet --trainer mas_caf --lamb 0.5 --lamb_kld 1e-3 --lamb_af 1e-8 --tasknum 10 --seed 0

#RWALK
python3 ./main.py --dataset tinyImageNet --trainer rwalk --lamb 5 --tasknum 10 --seed 0

#SI
python3 ./main.py --dataset tinyImageNet --trainer si --lamb 0.8 --tasknum 10 --seed 0

#AGS-CL
python3 ./main.py --experiment tinyImageNet --approach gs --lamb 0.1 --mu 0.001 --rho 0.1 --eta 0.9 --tasknum 10 --seed 0


#CORe50 (Results in Fig.5)

#EWC
python3 core50.py --save_path results/ --data_path data/ --cuda yes --n_epochs 40 --use 1.0 --model ewc --temperature 5 --lamb 1e2 --n_val 0.2 --data_file core50 --inner_steps 1 --n_meta 2 --n_memories 0 --lr 0.003 --beta 0.03 --n_runs 5 --batch_size 32 --n_tasks 10 --temperature 5

#EWC+CAF
python3 core50.py --save_path results/ --data_path data/ --cuda yes --n_epochs 40 --use 1.0 --model ewc_caf --temperature 5 --lamb 1e2 --lamb1 1e-1 --lamb2 1e-6 --n_val 0.2 --data_file core50 --inner_steps 1 --n_meta 2 --n_memories 0 --lr 0.003 --beta 0.03 --n_runs 5 --batch_size 32 --n_tasks 10 --temperature 5

#MAS
python3 core50.py --save_path results/ --data_path data/ --cuda yes --n_epochs 40 --use 1.0 --model mas --temperature 5 --lamb 1e-2 --n_val 0.2 --data_file core50 --inner_steps 1 --n_meta 2 --n_memories 0 --lr 0.003 --beta 0.03 --n_runs 5 --batch_size 32 --n_tasks 10 --temperature 5

MAS+CAF
python3 core50.py --save_path results/ --data_path data/ --cuda yes --n_epochs 40 --use 1.0 --model mas_caf --temperature 5 --lamb 1e-2 --lamb1 1e-1 --lamb2 1e-6 --n_val 0.2 --data_file core50 --inner_steps 1 --n_meta 2 --n_memories 0 --lr 0.003 --beta 0.03 --n_runs 5 --batch_size 32 --n_tasks 10 --temperature 5
