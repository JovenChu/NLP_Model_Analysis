# NLP_Model_Analysis
My imitate and any ideas for NLP Research



## Paper

1. [Bert](https://arxiv.org/abs/1810.04805)：

   * **BERT模型分析**

     BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，因为decoder是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。
     ![2019-08-21-021202](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-21-021652.png)

     - **模型结构**

       * 由于模型的构成元素Transformer已经解析过，就不多说了，BERT模型的结构如下图最左：

         ![img](https://pic1.zhimg.com/80/v2-d942b566bde7c44704b7d03a1b596c0c_hd.jpg)

       * 对比OpenAI GPT(Generative pre-trained transformer)，BERT是双向的Transformer block连接；就像单向rnn和双向rnn的区别，直觉上来讲效果会好一些。

       * 对比ELMo，虽然都是“双向”，但目标函数其实是不同的。ELMo是分别以![img](https://www.zhihu.com/equation?tex=P(w_i|+w_1%2C+...w_{i-1})) 和 ![img](https://www.zhihu.com/equation?tex=P(w_i|w_{i%2B1}%2C+...w_n)) 作为目标函数，独立训练处两个representation然后拼接，而BERT则是以 ![img](https://www.zhihu.com/equation?tex=P(w_i|w_1%2C++...%2Cw_{i-1}%2C+w_{i%2B1}%2C...%2Cw_n)) 作为目标函数训练LM。
       * 双向预测的例子说明：比如一个句子“BERT的新语言[mask]模型是“，遮住了其中的“表示”一次。双向预测就是用“BERT/的/新/语言/”（从前向后）和“模型/是”（从后向前）两种来进行bi-directional。但是在BERT当中，选用的是上下文全向预测[mask]，即使用“BERT/的/新/语言/.../模型/是”来预测，称为deep bi-directional。这就需要使用到Transformer模型来实现上下文全向预测，该模型的核心是聚焦机制，对于一个语句，可以同时启用多个聚焦点，而不必局限于从前往后的，或者从后往前的，序列串行处理。
       * 预训练 pre-training两个步骤：第一个步骤是把一篇文章中，15% 的词汇遮盖，让模型根据上下文全向地预测被遮盖的词。假如有 1 万篇文章，每篇文章平均有 100 个词汇，随机遮盖 15% 的词汇，模型的任务是正确地预测这 15 万个被遮盖的词汇。通过全向预测被遮盖住的词汇，来初步训练 Transformer 模型的参数。用第二个步骤继续训练模型的参数。譬如从上述 1 万篇文章中，挑选 20 万对语句，总共 40 万条语句。挑选语句对的时候，其中 20 万对语句，是连续的两条上下文语句，另外 20 万对语句，不是连续的语句。然后让 Transformer 模型来识别这 20 万对语句，哪些是连续的，哪些不连续。

     - **如何实现语言框架中的解析和组合**

       * 组合即是word由多个token组成。解析即通过对句子层次结构的拆解，可推导含义。这两个部分是Transformer极大程度需要依赖的两个操作，而且两者之间也是互相需要。

       * Transformer 通过迭代过程，连续的执行解析和合成步骤，以解决相互依赖的问题。Transformer 是由几个堆叠的层（也称为块）组成的。每个块由一个注意力层和其后的非线性函数（应用于 token）组成。

         ![image](http://ww4.sinaimg.cn/large/006tNc79gy1g60ik32rolj30cf05qaa8.jpg)

       *

     - **Embedding**

       * 这里的Embedding由三种Embedding求和而成：

         ![img](https://pic2.zhimg.com/80/v2-11505b394299037e999d12997e9d1789_hd.jpg)

       * 其中：

         * Token Embeddings是词向量，第一个单词是CLS标志，可以用于之后的分类任务
         * Segment Embeddings用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务
         * Position Embeddings和之前文章中的Transformer不一样，不是三角函数而是学习出来的

     - **Pre-training Task 1#: Masked LM**

       * 第一步预训练的目标就是做语言模型，从上文模型结构中看到了这个模型的不同，即bidirectional。**关于为什么要如此的bidirectional**，作者在[reddit](https://link.zhihu.com/?target=http%3A//www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_of_deep_bidirectional/)上做了解释，意思就是如果使用预训练模型处理其他任务，那人们想要的肯定不止某个词左边的信息，而是左右两边的信息。而考虑到这点的模型ELMo只是将left-to-right和right-to-left分别训练拼接起来。直觉上来讲我们其实想要一个deeply bidirectional的模型，但是普通的LM又无法做到，因为在训练时可能会“穿越”（**关于这点我不是很认同，之后会发文章讲一下如何做bidirectional LM**）。所以作者用了一个加mask的trick。

       * 在训练过程中作者随机mask 15%的token，而不是把像cbow一样把每个词都预测一遍。**最终的损失函数只计算被mask掉那个token。**

       * Mask如何做也是有技巧的，如果一直用标记[MASK]代替（在实际预测时是碰不到这个标记的）会影响模型，所以随机mask的时候10%的单词会被替代成其他单词，10%的单词不替换，剩下80%才被替换为[MASK]。具体为什么这么分配，作者没有说。。。要注意的是Masked LM预训练阶段模型是不知道真正被mask的是哪个词，所以模型每个词都要关注。

     - **Pre-training Task 2#: Next Sentence Prediction**

       - 因为涉及到QA和NLI之类的任务，增加了第二个预训练任务，目的是让模型理解两个句子之间的联系。训练的输入是句子A和B，B有一半的几率是A的下一句，输入这两个句子，模型预测B是不是A的下一句。预训练的时候可以达到97-98%的准确度。

       - **注意：作者特意说了语料的选取很关键，要选用document-level的而不是sentence-level的，这样可以具备抽象连续长序列特征的能力。**

     - **Fine-tunning**

       * 分类：对于sequence-level的分类任务，BERT直接取第一个[CLS]token的final hidden state ![img](https://www.zhihu.com/equation?tex=C%5Cin%5CRe%5EH) ，加一层权重 ![img](https://www.zhihu.com/equation?tex=W%5Cin%5CRe%5E%7BK%5Ctimes+H%7D) 后softmax预测label proba： ![img](https://www.zhihu.com/equation?tex=P%3Dsoftmax(CW^T)+\\)

       * 其他预测任务需要进行一些调整，如图：

         ![2019-08-21-021223](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-21-021624.png)

       * 可以调整的参数和取值范围有：
         * Batch size: 16, 32
         * Learning rate (Adam): 5e-5, 3e-5, 2e-5
         * Number of epochs: 3, 4

       因为大部分参数都和预训练时一样，精调会快一些，所以作者推荐多试一些参数。

   * **BERT优缺点**

     - **优点**

       BERT是截至2018年10月的最新state of the art模型，通过预训练和精调横扫了11项NLP任务，这首先就是最大的优点了。而且它还用的是Transformer，也就是相对rnn更加高效、能捕捉更长距离的依赖。对比起之前的预训练模型，它捕捉到的是真正意义上的bidirectional context信息。

     - **缺点**

       作者在文中主要提到的就是MLM预训练时的mask问题：

       - [MASK]标记在实际预测中不会出现，训练时用过多[MASK]影响模型表现
       - 每个batch只有15%的token被预测，所以BERT收敛得比left-to-right模型要慢（它们会预测每个token）

   * **总结**

     【参考资料】：

     1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1810.04805.pdf)
2. [全面超越人类！Google称霸SQuAD，BERT横扫11大NLP测试](https://zhuanlan.zhihu.com/p/46648916)
     3. [知乎：如何评价BERT模型？](https://www.zhihu.com/question/298203515?from=timeline&isappinstalled=0&utm_medium=social&utm_source=wechat_session)
     4. XLA加速：XLA是Tensorflow新近提出的模型编译器，其可以将Graph编译成IR表示，Fuse冗余Ops，并对Ops做了性能优化、适配硬件资源。然而官方的Tensorflow release并不支持xla的分布式训练，为了保证分布式训练可以正常进行和精度，我们自己编译了带有额外patch的tensorflow来支持分布式训练，Perseus-BERT 通过启用XLA编译优化加速训练过程并增加了Batch size大小。tensorflow中的图上的节点称之为operations或者ops。每个赋值、循环等计算操作都算是一个节点。
   
   

2. others：







## Code

1. Bert:

   * Build the environment：

     - Create environment：

       ```shell
       $ conda create -n bert python=3.6
       $ source activate bert
       ```

     - Tensorflow：

       ```shell
       $ pip install tensorflow # When you only use cpu to fine tune.Must >=1.11.0.
       $ pip install tensorflow-gpu # Using GPU to fine tune.Must match your CUDA version.
       ```

     - Collections：提供`namedtuple`、`deque`、`defaultdict`、`OrdereDict`、`Counter`等的方法，用于tuple、list、dict等删减，以及字符数量统计。

       ```shell
       $ pip isntall collections
       ```

       


   - Create pertraining data：
     - Class Training Instance:对单个句子的训练实例
       - setting the parameter：
         - instances , tokenizer , max_seq_length, max_predictions_per_seq, output_file
         - masked_lm_positions：被遮盖的词的位置
         - max_seq_length：最大序列（样本句子）长度
         - max_predictions_per_seq：每个序列（样本句子）中被遮盖的最大词长
       
     
   - Analysis the modeling：

     * Input the data:

       * Parameter setting:

         * guid: Unique id, 样本的唯一标识
         * tesxt_a：untokenized text, 未分词的序列文本。在单一序列任务中，仅text_a参数不能为空。
         * text_b：与text_a类似，用于序列（句子）对的任务中不能为空。
         * label：序列样本的标签，在train/evaluation中不能为空，predict任务中可以为空。

       * DataProcess class:

         * get_train_examples(self,fata_dir)、get_dev_examples()、get_test_examples()：需要有三个读取csv、tsv、txt等的函数，分别对应train、eval和predict三种模式。返回的是create_example()方法得到的样本列表
         * get_label()：定义任务的标签种类
         * create_example()：将数据中的id、text、label录入进入列表`example`中，以此完成数据的初始化。

       * Shuffle data：`d = d.shuffle(buffer_size=100)` 设置数据的扰乱系数，从而避免训练时使用单一label的文本进行不平衡训练。
       
       * 接下来需要处理数据以适合bert进行训练。步骤依次如下：
         - 单词全部小写
         - 将文本转换成序列（如：‘sally says hi’ -> ['sally','says','hi']）
         - 将单词分解为wordpieces（如：‘calling’->['call','##ing']）
         - 用bert提供的词汇文件进行单词索引映射
         - 添加‘CLS’,'SEP'标记符
         - 每次输入添加‘index’和‘segment’标记

     * Convert example to feature:

       * file_based_convert_examples_to_features()：作用是遍历examples列表，将单个的example转换成适用于bert的特征表示

       * BERT的特征表示形式：

         ```
         (a) For sequence pairs（句子对）:
         tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
         type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
         
         (b) For single sequences（单一文本）:
         tokens:   [CLS] the dog is hairy . [SEP]
         type_ids: 0     0   0   0  0     0 0
         ```

       * 输入输出：

         * 输入：examples = get_train_examples()、label_list = get_label()
         * 输出：feature = InputFeatures(input_ids,input_mask,segment_ids,label_id,is_real_example=True)

       * 最后将examples、labels、input_ids、input_mask、segment_ids、features等写入到模型输出路径的`output/train.tf_record`文件当中。**该文件包含模型训练所需的所有特征和参数。**

     * Tokenization for processing sequence to token：

       - 初始化并获取分割sequence成token的接口，in `tokenization.py`

         ```python
         tokenizer = tokenization.FullTokenizer(
             vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
         ```

       - 输入输出：

         - 输入：vocab_file（bert模型中的id2embedding词表）、do_lower_case（是否忽略大小写，默认为True）
         - 输出：tokenizer（基于bert词表的token分割器）

     * Training：
       * tf.contrib.tpu.TPUEstimator.train()：
       * 输入：train_file = “train.tf_record”、max_steps（训练步长=（样本数/batch_size * epoch））
       * 输出：checkpoint 模型文件

         

   - Squad:

     

   - Sequence pair:

2. Faster Transformer




