if [ $1 = 'train' ]; then
    echo "[!] train "
    rm exp/$2/*

    rm samples/sample.txt

    CUDA_VISIBLE_DEVICES="$3" python train.py \
        -s trs_small \
        --device cuda:0 \
        --mode train \
        --model_dataset $2 \
        --seed 30 \
        --n_vocab 20000 \
        --batch_size 64 

elif [ $1 = 'test' ]; then
    echo "[!] translate and evaluate"
    CUDA_VISIBLE_DEVICES="$3" python train.py \
        -s trs_small \
        --device cuda:0 \
        --mode test \
        --model_dataset $2 \
        --n_vocab 20000 \
        --batch_size 64 \
        --seed 30
elif [ $1 = 'eval' ]; then
    echo "[!] evaluate performance"
    python train.py --mode eval
else
    echo "[!] wrong mode $1"
fi
