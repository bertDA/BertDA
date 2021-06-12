### change these variables if needed
DATA_DIR=data
TASK_NAME=firstlevel
MODEL_TYPE=bert
MODEL_NAME=allenai/scibert_scivocab_uncased
SEED=300
OUTPUT=models/$SEED/$TASK_NAME/base
### end

python -m src.train \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --data_dir $DATA_DIR/$TASK_NAME \
    --max_seq_length 512\
    --learning_rate 2e-5 \
    --num_train_epochs 6.0 \
    --output_dir $OUTPUT \
    --prev_output_dir $OUTPUT\
    --seed $SEED \
    --base_model $MODEL_NAME


