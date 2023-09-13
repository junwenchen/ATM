

GPU=$1
save_dir=data/save_models/nextqa/atm_pt/
echo $save_dir
CUDA_VISIBLE_DEVICES=$GPU python main.py \
	--checkpoint_dir=nextqa \
	--dataset=nextqa \
	--mc=5 \
	--bnum=5 \
	--epochs=20 \
	--lr=0.00001 \
	--qmax_words=0 \
	--amax_words=38 \
	--max_feats=32 \
	--batch_size=64 \
	--batch_size_val=64 \
	--num_thread_reader=8 \
	--mlm_prob=0 \
	--n_layers=1 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--dropout=0.3 \
	--seed=666 \
	--flow_id='v4_n16' \
	--vis_path='/localscratch/vgt' \
	--sr=1 \
	--fusion=pesaca \
	--loss_list csl \
	--query_list qv \
	--csl_type=CE \
	--save_dir=${save_dir} 