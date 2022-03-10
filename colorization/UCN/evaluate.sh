# export CUDA_VISIBLE_DEVICES=3
python training/run_model_transformer_output.py \
    --input_dir /home/cain/data/toei/toei_test_conf \
    --checkpoint weights_transformer/model_040.pth \
    --output_dir outputs_conf_color_transformer_nonrule

python training/evaluate.py \
    --gt_dir /home/cain/data/toei/toei_test_conf \
    --out_dir outputs_conf_color_transformer_nonrule