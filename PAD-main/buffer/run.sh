python buffer_CL.py --dataset=MNIST --model=ConvNet --train_epochs=10 --num_experts=10 \
                    --zca --buffer_path=../buffer_storage/ --data_path=../dataset/ \
                    --sort_method="CIFAR10_GraNd" --rho_max=0.01 --rho_min=0.01 --alpha=0.3 \
                    --lr_teacher=0.01 --mom=0. --batch_train=256 --init_ratio=0.75 --add_end_epoch=20 \
                    --rm_epoch_first=40 --rm_epoch_second=60 --rm_easy_ratio_first=0.1 \
                    --rm_easy_ratio_second=0.2 