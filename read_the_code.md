已知问题
这个模型的sed是靠按帧分类得到的
其实也可以考虑这么干


程序入口
pytorch/inference.py - 简单的infer模块
pytorch/finetune_template.py - finetune的模板，还需要额外实现
pytorch/main.py - 完全实现的训练模块

拆解内容
数据集 - 数据是什么样子的，数据组织是什么样子的，加载方式又是如何？
 - 似乎需要打包成h5文件
[x] 用的是utils/dataset.py，调用入口在scripts\2_pack_waveforms_to_hdf5s.sh
[x] create training indice是在干嘛
[ ] CNN14在推理的时候，是如何处理不同长度输入的问题的

>> 尝试单步调试
python utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path="K:/t2/aud_debug/balanced_train_segments.csv" \
    --audios_dir="K:/t2/aud_debug/balanced_train_segments" \
    --waveforms_hdf5_path="K:/t2/aud_debug/balanced_train.h5" \
    --mini_data

简直有毒
取数据 - 每段音频取的10秒
关键是只取的开头10秒(pad or truncate)，还不是粗标注的时间范围
假定有num个sample，c个class
waveform加载进来，编码到h5文件里面，用的librosa加载的。target是[num, c]的数组

create training indice - 因为train文件有多个h5，相当于做了个大的索引把它们包在一起


模型
pytorch conv: BCHW

```CNN14
mel out: torch.Size([1, 1, 701, 64])
conv1 out: torch.Size([1, 64, 350, 32])
conv2 out: torch.Size([1, 128, 175, 16])
conv3 out: torch.Size([1, 256, 87, 8])
conv4 out: torch.Size([1, 512, 43, 4])
conv5 out: torch.Size([1, 1024, 21, 2])
conv6 out: torch.Size([1, 2048, 21, 2])
conv mean out: torch.Size([1, 2048, 21])  <  这个地方把维度给降下来了, 这样进出就一致了
add mean max out: torch.Size([1, 2048])
fc1 out: torch.Size([1, 2048])
clip out: torch.Size([1, 527])
```

```SED
mel out: torch.Size([1, 1, 701, 64])
conv1 out: torch.Size([1, 64, 350, 32])
conv2 out: torch.Size([1, 128, 175, 16])
conv3 out: torch.Size([1, 256, 87, 8])
conv4 out: torch.Size([1, 512, 43, 4])
conv5 out: torch.Size([1, 1024, 21, 2])
conv6 out: torch.Size([1, 2048, 21, 2])
conv mean out: torch.Size([1, 2048, 21])   < Batch, feat, 时间相关
add mean max out: torch.Size([1, 2048, 21])
fc1 out: torch.Size([1, 21, 2048])   < Batch, 时间相关, embedding
segment out: torch.Size([1, 21, 527])   < Batch, 时间相关, types - 一个segment差不多1s
clip out: torch.Size([1, 527])
frame out: torch.Size([1, 701, 527])  < segment到frame 的时候，直接duplicate结果
```

训练器


迁移训练方法

TODO: 
[x] 拆掉CNN14里面的melgram，换成外面扔进去的
[x] 研究拆除逻辑：经验证，直接把melspectrogram扔进去就行
[x] 搞清楚audio_tagging和sed两个任务的head是怎么设计的
[ ] 基于一个音频，搞一个数据集出来