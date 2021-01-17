import os 

# In-domain experiments
for out_domain in ['Laptops', 'Restaurants', 'Menstshirt', 'Television']:
	for in_domain in ['Laptops', 'Restaurants', 'Menstshirt', 'Television']:
		if in_domain != out_domain:
			continue
		for seed_val in [1, 2, 3, 4, 5]:			
			os.system(f'python train.py --model_name atae_lstm --train_dataset {out_domain} --test_dataset {in_domain} --batch_size 25 --l2reg 0.001 --valset_ratio 0.15 --seed {seed_val} --expr_idx {seed_val}')
			os.system(f'python train.py --model_name ram --train_dataset {out_domain} --test_dataset {in_domain} --l2reg 0.001 --learning_rate 0.005 --dropout 0.5 --valset_ratio 0.5 --num_epoch 15 --seed {seed_val} --expr_idx {seed_val}')
			os.system(f'python train.py --model_name ian --train_dataset {out_domain} --test_dataset {in_domain} --dropout 0.5 --l2reg 0.00001 --valset_ratio 0.15 --seed {seed_val} --expr_idx {seed_val}')
			os.system(f'python train.py --model_name bert_spc --batch_size 64 --train_dataset {out_domain} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0.1 --optimizer adam --initializer xavier_uniform_ --valset_ratio 0.15 --expr_idx {seed_val}')
			os.system(f'python train.py --model_name aen_bert --batch_size 32 --train_dataset {out_domain} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0.1 --optimizer adam --initializer xavier_uniform_ --valset_ratio 0.15  --expr_idx {seed_val}')
			os.system(f'python train.py --model_name lcf_bert --train_dataset {out_domain} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0 --local_context_focus cdw --optimizer adam --initializer xavier_uniform_ --batch_size 32 --valset_ratio 0.15  --expr_idx {seed_val}')


# Contrast experiments
for out_domain in ['Laptops', 'Restaurants', 'Menstshirt', 'Television']:
	for in_domain in ['Laptops', 'Restaurants', 'Menstshirt', 'Television']:
		if in_domain != out_domain:
			continue
		in_domain = in_domain + "_Contrast"
		for seed_val in [1, 2, 3, 4, 5]:			
			os.system(f'python train.py --model_name atae_lstm --train_dataset {out_domain} --test_dataset {in_domain} --batch_size 25 --l2reg 0.001 --valset_ratio 0.15 --seed {seed_val} --expr_idx {seed_val}')
			os.system(f'python train.py --model_name ram --train_dataset {out_domain} --test_dataset {in_domain} --l2reg 0.001 --learning_rate 0.005 --dropout 0.5 --valset_ratio 0.5 --num_epoch 15 --seed {seed_val} --expr_idx {seed_val}')
			os.system(f'python train.py --model_name ian --train_dataset {out_domain} --test_dataset {in_domain} --dropout 0.5 --l2reg 0.00001 --valset_ratio 0.15 --seed {seed_val} --expr_idx {seed_val}')
			os.system(f'python train.py --model_name bert_spc --batch_size 64 --train_dataset {out_domain} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0.1 --optimizer adam --initializer xavier_uniform_ --valset_ratio 0.15 --expr_idx {seed_val}')
			os.system(f'python train.py --model_name aen_bert --batch_size 32 --train_dataset {out_domain} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0.1 --optimizer adam --initializer xavier_uniform_ --valset_ratio 0.15  --expr_idx {seed_val}')
			os.system(f'python train.py --model_name lcf_bert --train_dataset {out_domain} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0 --local_context_focus cdw --optimizer adam --initializer xavier_uniform_ --batch_size 32 --valset_ratio 0.15  --expr_idx {seed_val}')


# Cross-domain experiments
for out_domain in ['Laptops', 'Restaurants', 'Menstshirt', 'Television']:
	for in_domain in ['Laptops', 'Restaurants', 'Menstshirt', 'Television']:
		if in_domain == out_domain:
			continue
		for seed_val in [1, 2, 3, 4, 5]:			
			os.system(f'python train.py --model_name atae_lstm --train_dataset {out_domain} --test_dataset {in_domain} --batch_size 25 --l2reg 0.001 --valset_ratio 0.15 --seed {seed_val} --expr_idx {seed_val}')
			os.system(f'python train.py --model_name ram --train_dataset {out_domain} --test_dataset {in_domain} --l2reg 0.001 --learning_rate 0.005 --dropout 0.5 --valset_ratio 0.5 --num_epoch 15 --seed {seed_val} --expr_idx {seed_val}')
			os.system(f'python train.py --model_name ian --train_dataset {out_domain} --test_dataset {in_domain} --dropout 0.5 --l2reg 0.00001 --valset_ratio 0.15 --seed {seed_val} --expr_idx {seed_val}')
			os.system(f'python train.py --model_name bert_spc --batch_size 64 --train_dataset {out_domain} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0.1 --optimizer adam --initializer xavier_uniform_ --valset_ratio 0.15 --expr_idx {seed_val}')
			os.system(f'python train.py --model_name aen_bert --batch_size 32 --train_dataset {out_domain} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0.1 --optimizer adam --initializer xavier_uniform_ --valset_ratio 0.15  --expr_idx {seed_val}')
			os.system(f'python train.py --model_name lcf_bert --train_dataset {out_domain} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0 --local_context_focus cdw --optimizer adam --initializer xavier_uniform_ --batch_size 32 --valset_ratio 0.15  --expr_idx {seed_val}')


# Incremental cross-domain experiments
for out_domain in ['Laptops', 'Restaurants', 'Menstshirt', 'Television']:
	for in_domain in ['Laptops', 'Restaurants', 'Menstshirt', 'Television']:
		if in_domain == out_domain:
			continue
		if out_domain == "Laptops" and in_domain != "Restaurants":			
			continue
		if out_domain == "Restaurants" and in_domain != "Laptops":			
			continue
		if out_domain == "Menstshirt" and in_domain != "Television":			
			continue
		if out_domain == "Television" and in_domain != "Menstshirt":			
			continue
		for ratio in [0.1, 0.25, 0.5]:
			train_file = out_domain + "_" + in_domain + "_" + str(ratio)
			for seed_val in [1, 2, 3, 4, 5]:				
				os.system(f'python train.py --model_name atae_lstm --train_dataset {train_file} --test_dataset {in_domain} --batch_size 25 --l2reg 0.001 --valset_ratio 0.15 --seed {seed_val} --expr_idx {seed_val}')
				os.system(f'python train.py --model_name ram --train_dataset {train_file} --test_dataset {in_domain} --l2reg 0.001 --learning_rate 0.005 --dropout 0.5 --valset_ratio 0.5 --num_epoch 15 --seed {seed_val} --expr_idx {seed_val}')
				os.system(f'python train.py --model_name ian --train_dataset {train_file} --test_dataset {in_domain} --dropout 0.5 --l2reg 0.00001 --valset_ratio 0.15 --seed {seed_val} --expr_idx {seed_val}')
				os.system(f'python train.py --model_name bert_spc --batch_size 64 --train_dataset {train_file} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0.1 --optimizer adam --initializer xavier_uniform_ --valset_ratio 0.15 --expr_idx {seed_val}')
				os.system(f'python train.py --model_name aen_bert --batch_size 32 --train_dataset {train_file} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0.1 --optimizer adam --initializer xavier_uniform_ --valset_ratio 0.15  --expr_idx {seed_val}')
				os.system(f'python train.py --model_name lcf_bert --train_dataset {train_file} --test_dataset {in_domain} --num_epoch 15 --seed {seed_val} --l2reg 1e-5 --dropout 0 --local_context_focus cdw --optimizer adam --initializer xavier_uniform_ --batch_size 32 --valset_ratio 0.15  --expr_idx {seed_val}')