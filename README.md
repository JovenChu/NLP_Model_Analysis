# NLP_Model_Analysis
My imitate and any ideas for NLP Research



## Paper

1. [Bert](https://arxiv.org/abs/1810.04805)：

   * **BERT模型分析**

     BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，因为decoder是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。

     ![2019-08-21-021202](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-21-021652.png)

     - **模型结构**

       * 由于模型的构成元素Transformer已经解析过，就不多说了，BERT模型的结构如下图最左：

         ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-21-021411.jpg)

       * 对比OpenAI GPT(Generative pre-trained transformer)，BERT是双向的Transformer block连接；就像单向rnn和双向rnn的区别，直觉上来讲效果会好一些。
   
       * 对比ELMo，虽然都是“双向”，但目标函数其实是不同的。ELMo是分别以![img](https://www.zhihu.com/equation?tex=P(w_i|+w_1%2C+...w_{i-1})) 和 ![img](https://www.zhihu.com/equation?tex=P(w_i|w_{i%2B1}%2C+...w_n)) 作为目标函数，独立训练处两个representation然后拼接，而BERT则是以 ![img](https://www.zhihu.com/equation?tex=P(w_i|w_1%2C++...%2Cw_{i-1}%2C+w_{i%2B1}%2C...%2Cw_n)) 作为目标函数训练LM。
     
    * 双向预测的例子说明：比如一个句子“BERT的新语言[mask]模型是“，遮住了其中的“表示”一次。双向预测就是用“BERT/的/新/语言/”（从前向后）和“模型/是”（从后向前）两种来进行bi-directional。但是在BERT当中，选用的是上下文全向预测[mask]，即使用“BERT/的/新/语言/.../模型/是”来预测，称为deep bi-directional。这就需要使用到Transformer模型来实现上下文全向预测，该模型的核心是聚焦机制，对于一个语句，可以同时启用多个聚焦点，而不必局限于从前往后的，或者从后往前的，序列串行处理。
   
       * 预训练 pre-training两个步骤：第一个步骤是把一篇文章中，15% 的词汇遮盖，让模型根据上下文全向地预测被遮盖的词。假如有 1 万篇文章，每篇文章平均有 100 个词汇，随机遮盖 15% 的词汇，模型的任务是正确地预测这 15 万个被遮盖的词汇。通过全向预测被遮盖住的词汇，来初步训练 Transformer 模型的参数。用第二个步骤继续训练模型的参数。譬如从上述 1 万篇文章中，挑选 20 万对语句，总共 40 万条语句。挑选语句对的时候，其中 20 万对语句，是连续的两条上下文语句，另外 20 万对语句，不是连续的语句。然后让 Transformer 模型来识别这 20 万对语句，哪些是连续的，哪些不连续。

     - **如何实现语言框架中的解析和组合**

       * 组合即是word由多个token组成。解析即通过对句子层次结构的拆解，可推导含义。这两个部分是Transformer极大程度需要依赖的两个操作，而且两者之间也是互相需要。

       * Transformer 通过迭代过程，连续的执行解析和合成步骤，以解决相互依赖的问题。Transformer 是由几个堆叠的层（也称为块）组成的。每个块由一个注意力层和其后的非线性函数（应用于 token）组成。

         ![image](http://ww4.sinaimg.cn/large/006tNc79gy1g60ik32rolj30cf05qaa8.jpg)

       * 注意力机制作为解析的步骤：

         * 注意力机制作用于序列（词或者token组成的句子）中，使得每个token注意到其他的token。

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-22-025714.png)

         * BERT中的每一层包含了12个独立的注意力头（注意力机制）。Google Research最近公开了BERT的张量流实现，并发布了以下预先训练的模型：
   
           1. `BERT-Base, Uncased`: 12层, 768个隐层, 12-heads, 110M 个参数
           2. `BERT-Large, Uncased`: 24层, 1024个隐层, 16-heads, 340M 个参数
           3. `BERT-Base, Cased`: 12层, 768个隐层, 12-heads , 110M 个参数
           4. `BERT-Large, Cased`: 24层, 1024个隐层, 16-heads, 340M 个参数
           5. `BERT-Base, Multilingual Cased (New, recommended)`: 104 种语言, 12层, 768个隐层, 12-heads, 110M 个参数
           6. `BERT-Base, Chinese`: Chinese Simplified and Traditional, 12层, 768个隐层, 12-heads, 110M 个参数
   
      * 由上可以看出BERT-Base模型中使用了12*12=144个注意力头：
   
        * 例句：we have grumpy neighbors if we keep the music up , they will get really angry.
   
        * 第二层的注意力头1，基于想换性形成组合成分。e.g. (get , angry) , (keep , up) and so on.
        * 第三层的注意力头11，token关注相同的中心词。e.g. (Keep、if、have)
        * 第五层注意力头6，匹配过程关注特定组合，发现动词组合等。e.g. (we, have), (if, we), (keep, up) (get, angry) 
        * 第六层注意力头0，解决指代消解。e.g. (they, neighbors)

      * 在每一层中，所有注意力头的输出被级接，并输入到一个可以表示复杂非线性函数的神经网络。

        ![image](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-22-055816.png)

      * **注意力机制的计算过程：**
    
           * 在某一个注意力头的作用下，会遍历**序列A**中每一个token元素，通过计算该token的query和**对比序列B**（可以是自身，可以是其他，视任务耳钉）中每个token的key矩阵的相似度（可通过点积、拼接等），然后通过softmax（加权求和）得到每个key对query的贡献度（概率分布）；
    
             ![image-20190822140910271](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-22-060911.png)
    
           * 然后使用这个贡献度做为权重，对value进行加权求和得到Attention的最终输出。在NLP中通常key和value是相同的。
    
             ![image-20190822140821506](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-22-060822.png)
    
           * 最终计算出每个token对该序列所有token的注意力得分，显示为可视化图像：
    
             ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-22-060012.png)
   
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
   
   
* **BERT优缺点**
  
  - **优点**
  
       - BERT是截至2018年10月的最新state of the art模型，通过预训练和精调横扫了11项NLP任务，这首先就是最大的优点了。而且它还用的是Transformer，也就是相对rnn更加高效、能捕捉更长距离的依赖。对比起之前的预训练模型，它捕捉到的是真正意义上的bidirectional context信息。
  
     - bert已经添加到TF-Hub模块，可以快速集成到现有项目中。bert层可以替代之前的elmo，glove层，并且通过fine-tuning，bert可以同时提供精度，训练速度的提升。
  
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
  
   ​    



2. XLnet:

   * 模型背景：

     * **使用自回归公式：**

       * AR（自回归，autoregressive）：

         * **原理：**自回归常见于时间序列分析或者信号处理领域。在语言模型中，被理解为一个句子的生成过程：首先根据概率分布生成第一个词，然后根据第一个词生成第二个词，然后根据前两个词生成第三个词，……，直到生成整个句子。

         * **优劣：**是一种是从上下文的词预测下一个词的语言模型。但是使用上下文的单词预测被限制在前向或后向，意味着它不能同时使用前向和后向的上下文信息预测。

         * **公式：**给定文本序列x=[x1,…,xT]，语言模型的目标是调整参数使得训练数据上的似然函数最大。

           * 记号x<t表示t时刻之前的所有x，也就是x1:t−1。hθ(x1:t−1)是RNN或者Transformer(注：Transformer也可以用于语言模型，比如在OpenAI GPT)编码的t时刻之前的隐状态。e(x)是词x的embedding。

             ![image-20190826114846530](/Users/jovenchu/Library/Application Support/typora-user-images/image-20190826114846530.png)

       * AE（自编码，autoencoding）的原理和优劣：

         * **原理：**自编码器是一种无监督学习输入的特征的方法：用一个神经网络把输入(输入通常还会增加一些噪声)变成一个低维的特征，这就是编码部分，然后再用一个Decoder尝试把特征恢复成原始的信号。BERT 就是一种自编码器语言模型，它通过 Mask 改变了部分 Token，然后试图通过其上下文的其它 Token 来恢复这些被 Mask 的 Token。

         * **优劣：**AE 的优势在于可以同时使用前向和后向的上下文来预测序列中被 [MASK] 替换的单词。缺点在于它假设预测（遮掩的）词在给定未遮掩的词的情况下彼此独立（独立性假设）。也就是当遮掩的词是邻近关系时，比如“旅游业危机”中的“旅游业”和“危机”，AE 模型会忽略掉“旅游业”和“危机”之间的关系，使得预测准确率降低。

         * **公式：**对于序列xx，BERT会随机挑选15%的Token变成[MASK]得到带噪声版本的x^。假设被Mask的原始值为x¯，那么BERT希望尽量根据上下文恢复(猜测)出原始值了。

           - 其中mt=1表示t时刻是一个Mask，需要恢复。Hθ是一个Transformer，它把长度为TT的序列xx映射为隐状态的序列Hθ(x)=[Hθ(x)1,Hθ(x)2,...,Hθ(x)T]。注意：前面的语言模型的RNN在t时刻只能看到之前的时刻，因此记号是hθ(x1:t−1)；而BERT的Transformer(不同与用于语言模型的Transformer)可以同时看到整个句子的所有Token，因此记号是Hθ(x)。

             ![image-20190826115002792](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-035003.png)

       * 两个模型的优缺点分别为：
         - **独立假设：**注意等式(2)的约等号≈≈，它的意思是假设在给定x^x^的条件下被Mask的词是独立的(没有关系的)，这个显然并不成立，比如”New York is a city”，假设Mask住”New”和”York”两个词，那么给定”is a city”的条件下”New”和”York”并不独立，因为”New York”是一个实体，看到”New”则后面出现”York”的概率要比看到”Old”后面出现”York”概率要大得多。而公式(1)没有这样的独立性假设，它是严格的等号。
         - **输入噪声：**BERT的在预训练时会出现特殊的[MASK]，但是它在下游的fine-tuning中不会出现，这就是出现了不匹配。而语言模型不会有这个问题。
         - **双向上下文：**
           - 语言模型只能参考一个方向的上下文，而BERT可以参考双向整个句子的上下文，因此这一点BERT更好一些。
           - ELMo和GPT最大的问题就是传统的语言模型是单向的——是根据之前的历史来预测当前词。但是不能利用后面的信息。比如句子”The animal didn’t cross the street because it was too tired”。在编码it的语义的时候需要同时利用前后的信息，因为在这个句子中，it可能指代animal也可能指代street。根据tired，推断它指代的是animal，因为street是不能tired。但是如果把tired改成wide，那么it就是指代street了。传统的语言模型，不管是RNN还是Transformer，它都只能利用单方向的信息。比如前向的RNN，在编码it的时候它看到了animal和street，但是它还没有看到tired，因此它不能确定it到底指代什么。如果是后向的RNN，在编码的时候它看到了tired，但是它还根本没看到animal，因此它也不能知道指代的是animal。Transformer的Self-Attention理论上是可以同时attend to到这两个词的，但是根据前面的介绍，由于需要用Transformer来学习语言模型，因此必须用Mask来让它看不到未来的信息，所以它也不能解决这个问题的。

     * **使用最大化预期可能性来学习上下文的分解顺序和排列：**

       * XLNet实现了一种泛化自回归方法 ，集合 AR 和 AE 的优点，避免两者的缺点。
       * XLNet 不使用传统 AR 模型中固定的前向或后向因式分解顺序，而是最大化所有可能因式分解顺序的期望对数似然。由于对因式分解顺序的排列操作，每个位置的语境都包含来自左侧和右侧的 token。因此，每个位置都能学习来自所有位置的语境信息，即捕捉双向语境。
       * 作为一个泛化 AR 语言模型，XLNet 不依赖残缺数据。因此，XLNet 不会有 BERT 的预训练-微调差异。同时，自回归目标提供一种自然的方式，来利用乘法法则对预测 token 的联合概率执行因式分解（factorize），这消除了 BERT 中的独立性假设。

     * **集成了Transformer-XL：**

       * 原理：

         * ![20190706164606164](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-062712.png)

         * 为了在内存的限制下让 Transformer 学到更长的依赖，Transformer-XL 借鉴了 TBPTT(Truncated Back-Propagation Through Time) 的思路，将上一个片段 s{t-1} 计算出来的表征缓存在内存里，加入到当前片段 s{t} 的表征计算中。
           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-062759.jpg)

         * 如上图所示，由于计算第 l 层的表征时，使用的第 l-1 层的表征同时来自于片段 s{t} 和 s{t-1}，所以每增加一层，模型建模的依赖关系长度就能增加 N。在上图中，Transformer-XL 建模的最长依赖关系为 3*2=6。

           但这又会引入新的问题。Transformer 的位置编码 (Position eEmbedding) 是绝对位置编码 (Absolute Position Embedding)，即每个片段内，各个位置都有其独立的一个位置编码向量。所以片段 s{t} 第一个词和片段 s{t-1} 第一个词共享同样的位置编码 -- 这会带来歧义。

           Transformer-XL 引入了更加优雅的相对位置编码 (Relative Position Embedding)。

         * 因为位置编码只在自注意力算子中起作用，将 Transformer 的自注意力权重的计算拆解成：

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-062913.png)

           可以将其中的绝对位置编码 p{j} 的计算替换成相对位置编码 r{i-j}，把 p{i} 替换成一个固定的向量 (认为位置 i 是相对位置的原点)。这样便得到相对位置编码下的注意力权重：

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-063016.png)

           Transformer-XL 的实际实现方式与上式有所不同，但思想是类似的。

           相对位置编码解决了不同片段间位置编码的歧义性。通过这种拆解，可以进一步将相对位置编码从词的表征中抽离，只在计算注意力权重的时候加入。这可以解决 Transformer 随着层数加深，输入的位置编码信息被过多的计算抹去的问题。Transformer-XL 在 XLNet 中的应用使得 XLNet 可以建模更长的依赖关系。

            ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-063155.png)

           <center><font size="2" color="gray">普通的Transformer语言模型的训练和预测</font></center>

           上图做是普通的Transformer语言模型的训练过程。假设Segment的长度为4，如图中我标示的：根据红色的路径，虽然x8的最上层是受x1影响的，但是由于固定的segment，x_8无法利用x1的信息。而预测的时候的上下文也是固定的4，比如预测x6时需要根据[x2,x3,x4,x5]来计算，接着把预测的结果作为下一个时刻的输入。接着预测x7的时候需要根据[x3,x4,x5,x6]完全进行重新的计算。之前的计算结果一点也用不上。

           ![image-20190826150346740](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-070346.png)

           <center><font size="2" color="gray">Transformer-XL的训练和预测</font></center>

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-063133.png)

       * XLNet 将 Transformer-XL 的分割循环机制（segment recurrence mechanism）和相对编码范式（relative encoding）整合到预训练中

       * 简单地使用 Transformer(-XL) 架构进行基于排列的（permutation-based）语言建模是不成功的，因为因式分解顺序是任意的、训练目标是模糊的。因此，研究人员提出，对 Transformer(-XL) 网络的参数化方式进行修改，移除模糊性。

     

   * 模型目标：排列语言建模（Permutation Language Modeling）：

     * 研究者借鉴了无序 NADE 中的想法，提出了一种序列语言建模目标，它不仅可以保留 AR 模型的优点，同时也允许模型捕获双向语境。具体来说，一个长度为 T 的序列 x 拥有 T! 种不同的排序方式，可以执行有效的自回归因式分解。如果模型参数在所有因式分解顺序中共享，那么预计模型将学习从两边的所有位置上收集信息。

     * 研究者展示了一个在给定相同输入序列 x（但因式分解顺序不同）时预测 token x_3 的示例，如下图所示：比如图的左上，对应的分解方式是3→2→4→13→2→4→1，因此预测x3是不能attend to任何其它词，只能根据之前的隐状态mem来预测。而对于左下，x3可以attend to其它3个词。

       ![image-20190826115137425](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-035138.png)

     * 给定长度为T的序列xx，总共有T!种排列方法，也就对应T!种链式分解方法。比如假设x=x1x2x3，那么总共用3!=6种分解方法：

       ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-035753.png)

       - p(x2|x1x3)指的是第一个词是x1并且第三个词是x3的条件下第二个词是x2的概率，也就是说原来词的顺序是保持的。如果理解为第一个词是x1并且第二个词是x3的条件下第三个词是x2，那么就不对了。

     * 遍历分析：

       * 如果的语言模型遍历T!种分解方法，并且这个模型的参数是共享的，那么这个模型应该就能(必须)学习到各种上下文。普通的从左到右或者从右往左的语言模型只能学习一种方向的依赖关系，比如先”猜”一个词，然后根据第一个词”猜”第二个词，根据前两个词”猜”第三个词，……。而排列语言模型会学习各种顺序的猜测方法，比如上面的最后一个式子对应的顺序3→1→2，它是先”猜”第三个词，然后根据第三个词猜测第一个词，最后根据第一个和第三个词猜测第二个词。

       * 因此可以遍历T!种路径，然后学习语言模型的参数，但是这个计算量非常大(10!=3628800,10个词的句子就有这么多种组合)。因此实际只能随机的采样T!里的部分排列，为了用数学语言描述，引入几个记号。ZT表示长度为T的序列的所有排列组成的集合，则z∈ZT是一种排列方法。用zt表示排列的第t个元素，而z<t表示z的第1到第t-1个元素。

       * 举个例子，假设T=3，那么ZT共有6个元素，假设其中之一z=[1,3,2]，则z3=2，而z<3=[1,3]。有了上面的记号，则排列语言模型的目标是调整模型参数使得下面的似然概率最大：

         ![image-20190826120103895](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-040104.png)

         * 上面的公式看起来有点复杂，细读起来其实很简单：从所有的排列中采样一种，然后根据这个排列来分解联合概率成条件概率的乘积，然后加起来。

         * 注意：上面的模型只会遍历概率的分解顺序，并不会改变原始词的顺序。实现是通过Attention的Mask来对应不同的分解方法。比如p(x1|x3)p(x2|x1x3)p(x3)，可以在用Transformer编码x1时候让它可以Attend to x3，而把x2Mask掉；编码x3的时候把x1,x2都Mask掉。
           

   * 模型架构：对目标感知表征的双流自注意力：

     * **架构图：**

       ![image-20190826151059909](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-071100.png)

       <center><font size="2" color="gray">Two-stream 排列模型的计算过程</font></center>

       ![20190706163804429](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-072144.png)

     * **双流（two-stream）：**（1）Content stream attention；（2）Query stream sttention.

     * **Content流Attention计算：**

       * 词向量序列作为输入，表征序列作为输出。下图中记 "MASK" 对应的词向量为 G，X2 - X4 为各自的词向量，G1, H1 - H4 为各自的表征。图中省略了位置编码 p。

         ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-082012.jpg)

       * 假设整句话为 ["我 1", "今天 2", "很 3",「开心 4」]，只采样出一个样本 (["今天 2", "很 3", "开心 4"] → "我 1" )，XLNet 是将目标词 "我 1" 替换成一个特殊字符 "MASK1"。和 BERT 不同，"MASK" 不会纳入表征的地址向量 k 以及内容向量 v 的计算，"MASK" 自始至终只充当了查询向量 q 的角色，因此所有词的表征中都不会拿到 "MASK" 的信息。这也杜绝了 "MASK" 的引入带来的预训练-微调差异 (Pretrain-Finetune Discrepancy)。

     * **双通道自注意力：**

       * 功能：实现同时计算两套表征：内容表征通道 (Content Stream) h 和语境表征通道 (Query Stream) g。

       * 提出背景：为了保证训练效率，需要只进行一次整句的表征计算便可以获得所有样本中的语境特征。这时所有词的表征就必须同时计算，此时便有标签泄露带来的矛盾：对于某个需要预测的目标词，既需要得到包含它信息以及位置的表征 h (用来进一步计算其他词的表征)，又需要得到不包含它信息，只包含它位置的表征 g (用来做语境的表征)。

       * 解决方案：

         * 假设要计算第 1 个词在第 l 层的语境表征 g{1}^{l} 和内容表征 h{1}^{l}，只关注注意力算子查询向量 Q、地址向量 K 以及内容向量 V 的来源：

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-082955.png)

         * 计算 g{1}^l 时用到了 h{j!=1}^{l-1}，表示第 l-1 层除了第 1 个词外所有词的表征，这是为了保证标签不泄露（计算位置表征时不用到标签内容）；计算 h{1}^{l} 时用到了 h{:}^{l-1}，表示第 l-1 层所有词的表征，这和标准的 Transformer 计算表征的过程一致。

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-083020.jpg)

         * 但是在两层自注意力算子的计算中可以看出，第 l-2 层第 1 个词的表征 h{1}^{l-2} 会通过第 l-1 层的所有表征 h{j}^{l-1} 泄露给 g{1}^{l}。

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-084202.png)

         * 需要对每层的注意力使用注意力掩码 (Attention Mask)，根据选定的分解排列 z，将不合理的注意力权重置零。我们记 z{t} 为分解排列中的第 t 个词，那在词 z{t} 的表征时，g{t}^{l} 和 h{t}^{l} 分别只能看到排列中前 t-1 个词 z{1:t-1} 和前 t 个词 z{1:t}，即

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-084234.png)

         * 在如此做完注意力掩码后，所有 g{z{t}}^l 便可以直接用来预测词 z{t}，而不会有标签泄露的问题。在具体实现效率的限制下，想要获得多样的语境并防止标签泄露，只能依据乱序语言模型的定义去使用注意力掩码。

       * 举例：

         * 假设输入的句子是”I like New York”，并且一种排列为z=[1, 3, 4, 2]，假设我们需要预测z3=4，那么根据公式：

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-084421.png)

         * 比如x，我们假设x是”York”，则pθ(X4=x)表示第4个词是York的概率。用自然语言描述：上面的概率是**第一个词是I，第3个词是New的条件下第4个词是York的概率**。

         * 另外我们再假设一种排列为z’=[1,3,2,4]，我们需要预测z3=2，那么：

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-084800.png)

         * 则上面是表示是**第一个词是I，第3个词是New的条件下第2个词是York的概率**。仔细对比一下公式会发现这两个概率是相等的。但是根据经验，显然这两个概率是不同的，而且上面的那个概率大一些，因为New York是一个城市。

         * 出现相等的问题关键是模型并不知道要预测的那个词在原始序列中的位置。位置编码是和输入的Embedding加到一起作为输入的，因此pθ(X4=x|x1x3)里的x1和x3是带了位置信息的，模型(可能)知道(根据输入的向量猜测)I是第一个词，而New是第三个词，但是第四个词的向量显然这个是还不知道(知道了还要就不用预测了)，因此就不可能知道它要预测的词到底是哪个位置的词，因此必须”显式”的告诉模型我要预测哪个位置的词。

         * 给定排列z，需要计算![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-085041.png)，如果使用普通的Transformer，那么计算公式为：

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-085058.png)

         * 为了解决“模型并不知道要预测的到底是哪个位置的词”这个问题，把预测的位置zt放到模型里：

           ![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-085208.png)

         * 上式中![img](https://img-blog.csdnimg.cn/20190706162737354.png)表示这是一个新的模型g，并且它的参数除了之前的词![img](https://img-blog.csdnimg.cn/20190706162756725.png)，还有要预测的词的位置![img](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-085225.png)

       

   * 多段建模：

     * 建模多个segment：

       * 许多下游的任务会有多余一个输入序列，比如问答的输入是问题和包含答案的段落。下面我们讨论怎么在自回归框架下怎么预训练两个segment。和BERT一样，我们选择两个句子，它们有50%的概率是连续的句子(前后语义相关)，有50%的概率是不连续(无关)的句子。我们把这两个句子拼接后当成一个句子来学习排列语言模型。输入和BERT是类似的：[A, SEP, B, SEP, CLS]，这里SEP和CLS是特殊的两个Token，而A和B代表两个Segment。而BERT稍微不同，这里把CLS放到了最后。原因是因为对于BERT来说，Self-Attention唯一能够感知位置是因为我们把位置信息编码到输入向量了，Self-Attention的计算本身不考虑位置信息。而前面我们讨论过，为了减少计算量，这里的排列语言模型通常只预测最后1/K个Token。我们希望CLS编码所有两个Segment的语义，因此希望它是被预测的对象，因此放到最后肯定是会被预测的。

       * 但是和BERT不同，我们并没有增加一个预测下一个句子的Task，原因是通过实验分析这个Task加进去后并不是总有帮助。【注：其实很多做法都是某些作者的经验，后面很多作者一看某个模型好，那么所有的Follow，其实也不见得就一定好。有的时候可能只是对某个数据集有效果，或者效果好是其它因素带来的，一篇文章修改了5个因素，其实可能只是某一两个因素是真正带来提高的地方，其它3个因素可能并不有用甚至还是有少量副作用。】

     * 相对segment编码：

       * BERT使用的是绝对的Sgement编码，也就是第一个句子对于的Segment id 是0，而第二个句子是1。这样如果把两个句子换一下顺序，那么输出就是不一样的了。XLNet使用的是相对的Segment编码，它是在计算Attention的时候判断两个词是否属于同一个Segment，如果位置 i 和位置 j 的词属于同一个Segment，那么使用一个可以学习的Embedding![image-20190826173848843](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-093849.png)，否则![image-20190826174158884](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-094159.png)，其中![image-20190826174333276](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-094333.png)是每个注意头的可学习模型参数。也就是说，只关心他们是属于同一个Segment还是属于不同的Segment。当我们从位置 i attention 到 j，我们会这样计算一个新的注意权重![image-20190826174223659](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-094224.png)，其中![image-20190826174254565](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-094254.png)是第 i 个位置的Query 向量，b是可学习的头部特异性偏向量（bias）。最后，将值![image-20190826174314433](https://joven-1252328025.cos.ap-shanghai.myqcloud.com/2019-08-26-094315.png)添加到正常的注意力。
       * 使用相对分段编码有两个好处。首先，相对编码的归纳偏差改善了泛化。其次，它打开了对具有两个以上输入段的任务进行微调的可能性。
3. 



## Code

1. Bert：

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
         - Key logic:
       
    - Text_Classifier：
     
      * Input the data:
 
        * Parameter setting:
          * guid: Unique id, 样本的唯一标识
          * tesxt_a：untokenized text, 未分词的序列文本。在单一序列任务中，仅text_a参数不能为空。
          * text_b：与text_a类似，用于序列（句子）对的任务中不能为空。用于句子关系判断（问答、翻译等。）
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
          
      - Attention：
     
          * 源码在`modeling.py`中实现。
     
          * transformer_model()：构建Transformer模型，在模型中加入注意力机制。获取预训练模型所训练所得的参数：hidden_size（隐藏层个数）, num_hidden_layers（层数）, num_attention_heads（注意力头数量）, attention_probs_dropout_prob
     
          * 计算注意力头大小：attention_head_size = int(hidden_size / num_attention_heads)
     
          * attention_layer() ：实现注意力机制计算。
     
            * transpose_for_scores()：计算张量矩阵的转置函数。
     
            * query_layer、key_layer、value_layer：实现了基于序列token特征到Q、K、V三个变量的计算。并使用transpose_for_scores()进行张量转置。
     
            * 计算注意力得分：
     
              ```python
              # Take the dot product between "query" and "key" to get the raw
              attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
              attention_scores = tf.multiply(attention_scores,
                                               1.0 / math.sqrt(float(size_per_head)))
              ```
     
            * 计算attention_scores的归一化概率，并进行dropout运算：
     
              ```python
              # Normalize the attention scores to probabilities.
              # `attention_probs` = [B, N, F, T]
              attention_probs = tf.nn.softmax(attention_scores)
              # This is actually dropping out entire tokens to attend to, which might
              # seem a bit unusual, but is taken from the original Transformer paper.
              attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
              ```
     
            * 输出：context_layer = tf.matmul(attention_probs, value_layer)
     
            * 使用：attention_output = attention_heads[0]
          
        - Squad:
     
        - Sequence pair:

2. Faster Transformer：




