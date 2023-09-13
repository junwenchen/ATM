# # 58.27, with shuffle
# GPU=$1
# for lambda in 0.15; do
#   for fusion in pesaca; do
# 	for csl_type in CE; do
# 		save_dir=../data/save_models/nextqa/VGT_B5_motv1_appk_clip{}_cl1{}_cl2{}_sr{}_${fusion}_${csl_type}_${lambda}_vqa_qv_shuffle_qtype_nclip_cvpr_sc_1/
# 		echo $save_dir
# 		CUDA_VISIBLE_DEVICES=$GPU python main.py \
# 			--checkpoint_dir=nextqa \
# 			--dataset=nextqa \
# 			--mc=5 \
# 			--bnum=5 \
# 			--epochs=20 \
# 			--lr=0.000005 \
# 			--qmax_words=0 \
# 			--amax_words=38 \
# 			--max_feats=32 \
# 			--batch_size=64 \
# 			--batch_size_val=64 \
# 			--num_thread_reader=8 \
# 			--mlm_prob=0 \
# 			--n_layers=1 \
# 			--embd_dim=512 \
# 			--ff_dim=1024 \
# 			--dropout=0.3 \
# 			--seed=666 \
# 			--flow_id='v4_n16' \
# 			--vis_path='/localscratch/vgt' \
# 			--sr=1 \
# 			--fusion=${fusion} \
# 			--loss_list "vqa" "shuffle" \
#     		--query_list qv \
# 			--lambda_sf=${lambda} \
# 			--csl_type=${csl_type} \
# 			--save_dir=${save_dir} \
# 			--pretrain_path=../data/save_models/nextqa/VGT_B5_motv1_appk_clipv4_n16_cl1False_cl2False_sr1_pesaca_CE_csl_qv/best_model_csl.pth
# 	done
#   done
# done

GPU=$1
save_dir=data/save_models/nextqa/atm_ft/
echo $save_dir
CUDA_VISIBLE_DEVICES=$GPU python main.py \
	--checkpoint_dir=nextqa \
	--dataset=nextqa \
	--mc=5 \
	--bnum=5 \
	--epochs=20 \
	--lr=0.000005 \
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
	--loss_list "vqa" "shuffle" \
	--query_list qv \
	--lambda_sf=0.15 \
	--csl_type=CE \
	--save_dir=${save_dir} \
	--pretrain_path=data/models/best_model_csl.pth