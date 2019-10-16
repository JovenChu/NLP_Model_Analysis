## ALBert

- 结果总结：

  - 参数设置：
    - max_seq_length：128
    - train_batch_size：16
    - eval_batch_size：16
    - predict_batch_size：16
    - learning_rate：5e-5
    - num_train_epochs：3.0
    - save_checkpoints_steps：100
  - 结果对比：

  | Task classification | Global_step | Eval_accuracy | Eval_loss | Loss | Samples |
  | :-----------------: | :---------: | :-----------: | :-------: | ---- | :-----: |
  |    ALBert_LCQMC     |    3749     |     0.83      |   0.41    | 0.41 |   2w    |
  |     Bert_LCQMC      |    3749     |     0.51      |   0.79    | 0.79 |   2w    |

  <font size="2" color="gray">Note: The experimental configuration is 11G Nvidia RTX2080Ti, Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz, 16G RAM, 2T hard disk</font>

- Albert：

  * 代码：

    ```shell
    $ export BERT_BASE_DIR=./albert/albert_large_zh
    $ export TEXT_DIR=./data/LCQMC
    $ python run_classifier.py   --task_name=lcqmc_pair   --do_train=true   --do_eval=true   --data_dir=$TEXT_DIR   --vocab_file=./albert_config/vocab.txt  \
            --bert_config_file=./albert_config/albert_config_large.json --max_seq_length=128 --train_batch_size=16   --learning_rate=2e-5  --num_train_epochs=3 \
            --output_dir=albert_large_lcqmc_checkpoints --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt
    ```

  * 结果：

    * LCQMC：

  ![image-20191015120247317](https://tva1.sinaimg.cn/large/006y8mN6gy1g7yr4ocap3j30h701ywem.jpg)

- bert

  * 代码：

    ```shell
    $ export BERT_BASE_DIR='./chinese_L-12_H-768_A-12'
    $ export LCQMC_DIR='./data/LCQMC'
    
    $ python run_classifier.py   --task_name=Lcqmc   --do_train=true   --do_eval=true   --do_predict=true   --data_dir=$LCQMC_DIR   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --max_seq_length=128   --train_batch_size=16   --output_dir=lcqmc_output
    
    $ python run_classifier.py   --task_name=Lcqmc   --do_eval=true   --data_dir=$LCQMC_DIR   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=lcqmc_output/model.ckpt-3749   --max_seq_length=128   --eval_batch_size=16   --output_dir=lcqmc_output
    ```

  * 结果：

    * LCQMC：

  ![image-20191015141907564](https://tva1.sinaimg.cn/large/006y8mN6gy1g7yv2ilx2uj30a2020jrc.jpg)

* 补充说明：

  1. 暂时找不到**ALBERT的官方版本开源预训练模型**，只有一个tensorflow的**中文**预训练模型开源，官方的论文中没有提到，也没有说明何时开源。**所以BERT中的所有英文数据集，无法在ALBert中对比实验！！！**

  2. 先用LCQMC中2w的样本训练作对比测试，完整的数据集训练下文有结果展示：https://github.com/brightmart/albert_zh

     * ALBert中文任务集上效果对比测试

     ### 自然语言推断：XNLI of Chinese Version

     | 模型             | 开发集      | 测试集      |
     | ---------------- | ----------- | ----------- |
     | BERT             | 77.8 (77.4) | 77.8 (77.5) |
     | ERNIE            | 79.7 (79.4) | 78.6 (78.2) |
     | BERT-wwm         | 79.0 (78.4) | 78.2 (78.0) |
     | BERT-wwm-ext     | 79.4 (78.6) | 78.7 (78.3) |
     | XLNet            | 79.2        | 78.7        |
     | RoBERTa-zh-base  | 79.8        | 78.8        |
     | RoBERTa-zh-Large | 80.2 (80.0) | 79.9 (79.5) |
     | ALBERT-base      | 77.0        | 77.1        |
     | ALBERT-large     | 78.0        | 77.5        |
     | ALBERT-xlarge    | ?           | ?           |
     | ALBERT-xxlarge   | ?           | ?           |

     

     ### 问题匹配语任务：LCQMC(Sentence Pair Matching)

     | 模型                                | 开发集(Dev)      | 测试集(Test) |
     | ----------------------------------- | ---------------- | ------------ |
     | BERT                                | 89.4(88.4)       | 86.9(86.4)   |
     | ERNIE                               | 89.8 (89.6)      | 87.2 (87.0)  |
     | BERT-wwm                            | 89.4 (89.2)      | 87.0 (86.8)  |
     | BERT-wwm-ext                        | -                | -            |
     | RoBERTa-zh-base                     | 88.7             | 87.0         |
     | RoBERTa-zh-Large                    | ***89.9(89.6)*** | 87.2(86.7)   |
     | RoBERTa-zh-Large(20w_steps)         | 89.7             | 87.0         |
     | ALBERT-zh-base-additional-36k-steps | 87.8             | 86.3         |
     | ALBERT-zh-base                      | 87.2             | 86.3         |
     | ALBERT-large                        | 88.7             | 87.1         |
     | ALBERT-xlarge                       | 87.3             | ***87.7***   |
     | ALBERT-xxlarge                      | ?                | ?            |

     

     ### 语言模型、文本段预测准确性、训练时间 Mask Language Model Accuarcy & Training Time

     | Model             | MLM eval acc | SOP eval acc | Training(Hours) | Loss eval |
     | ----------------- | ------------ | ------------ | --------------- | --------- |
     | albert_zh_base    | 79.1%        | 99.0%        | 6h              | 1.01      |
     | albert_zh_large   | 80.9%        | 98.6%        | 22.5h           | 0.93      |
     | albert_zh_xlarge  | ?            | ?            | 53h(预估)       | ?         |
     | albert_zh_xxlarge | ?            | ?            | 106h(预估)      | ?         |

     